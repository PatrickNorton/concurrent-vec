//! A wait-free vector based on a [paper][a] by Feldman, Valera-Leon, and
//! Dechev.
//!
//! It supports random-access reads and writes, push/pop, and
//! insert/remove operations.
//!
//! [a]: https://www.osti.gov/servlets/purl/1427291

mod data;
mod descr;
mod iter;

use crate::data::{Data, PartialData};
use crate::descr::{is_descr, Descriptor, Node, PopDescr, PushDescr, Value};
use crate::iter::{IntoIter, Iter};
use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};
use std::borrow::Borrow;
use std::fmt::{self, Debug, Formatter};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::{array, mem, ptr};

const LIMIT: usize = 32;

#[macro_export]
macro_rules! fvd_vec {
    () => {FvdVec::new()};

    ($($x:expr),+ $(,)?) => (
        FvdVec::from([$($x),+])
    );
}

/// A wait-free vector based on the paper by Feldman et al. It offers
/// random-access reads and writes, as well as push/pop operations.
pub struct FvdVec<T> {
    // SAFETY GUARANTEES:
    // * self.data is either null or points to a valid slice.
    // * * If self.data is null, this vector is empty.
    // * All values in the array are initialized, though maybe with null.
    // * All `Atomic<Value<T>>` that are non-null and non-`NOT_VALUE` follow
    //   the tag convention.
    data: Atomic<PartialData<T>>,
    length: AtomicUsize,
    phantom: PhantomData<T>,
}

pub struct Ref<'a, T> {
    _parent: &'a FvdVec<T>,
    value: Node<T>,
}

impl<T> FvdVec<T> {
    /// Creates a new, empty `WFVec<T>`.
    ///
    /// This method ought to be `const`, but there's no const way to create
    /// an `Atomic<T>` yet.
    pub fn new() -> FvdVec<T> {
        FvdVec {
            data: Atomic::null(),
            length: AtomicUsize::new(0),
            phantom: PhantomData,
        }
    }

    /// Creates a `WFVec<T>` with the given capacity.
    pub fn with_capacity(cap: usize) -> FvdVec<T> {
        let data = Owned::init(cap);
        FvdVec {
            data: Atomic::from(data),
            length: AtomicUsize::new(0),
            phantom: PhantomData,
        }
    }

    /// Gets the length of the vector.
    ///
    /// Because this vector is concurrent, the length may change in between
    /// when it is loaded and when it is used.
    pub fn len(&self) -> usize {
        self.length.load(Ordering::SeqCst)
    }

    /// Checks if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the current capacity of the vector.
    pub fn capacity(&self) -> usize {
        let guard = &epoch::pin();
        let data = self.data.load(Ordering::SeqCst, guard);
        if data.is_null() {
            0
        } else {
            // SAFETY: If `self.data` is not null, it is safe to load.
            unsafe { data.deref().len() }
        }
    }

    /// Reserves space for `additional` values to be inserted in `self`.
    pub fn reserve(&self, additional: usize) {
        // I'm not 100% sure we need the pin here, but let's keep it just in
        // case.
        let guard = &epoch::pin();
        self.get_spot(self.length.load(Ordering::SeqCst) + additional, &guard);
    }

    /// Appends an element to the end of the vector.
    pub fn push(&self, value: T) {
        let value = Atomic::new(Value::new_data(value));
        let mut pos = self.length.load(Ordering::SeqCst);
        let mut failures = 0;
        // SAFETY:
        //  * PushDescr is always a Value created from `Value::new_data`.
        //  * If the descriptor completes successfully, then we return
        //    immediately and the destructor is not called.
        let mut ph = unsafe { PushDescr::new_value(value.clone()) };
        let guard = &epoch::pin();
        loop {
            failures += 1;
            if failures > LIMIT {
                todo!("announce_op(PushOp(value))")
            }
            let mut spot = self.get_spot(pos, guard);
            let expected = spot.load(Ordering::SeqCst, guard);
            if expected.is_null() {
                if pos == 0 {
                    match spot.compare_exchange(
                        expected,
                        value.load(Ordering::SeqCst, guard),
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                        guard,
                    ) {
                        Result::Ok(_) => {
                            self.increment_size();
                            return;
                        }
                        Result::Err(_) => {
                            pos += 1;
                            spot = self.get_spot(pos, guard);
                        }
                    }
                }
                let value_desc = Value::new_descriptor(Descriptor::Push(ph));
                let owned = Owned::new(value_desc).with_tag(1);
                match spot.compare_exchange(
                    expected,
                    owned,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Result::Ok(desc) => {
                        // SAFETY: We know there's a descriptor in this and it
                        // follows the tag convention (b/c we just put it
                        // there).
                        let result = unsafe { Descriptor::complete_unchecked(desc, pos, &self) };
                        if result {
                            self.increment_size();
                            // SAFETY: The guard has fulfilled its purpose, so
                            // it is no longer accessible from the vector. As
                            // such, no other thread can get it after it is
                            // destroyed. Since we put the guard into the
                            // vector, we are the ones responsible for getting
                            // rid of it.
                            // NOTE: We can't use `value` from here on out,
                            // because the descriptor took it.
                            unsafe { Value::defer_drop(desc, guard) };
                            return;
                        } else {
                            pos -= 1;
                            // SAFETY: The guard has fulfilled its purpose, so
                            // it is no longer accessible from the vector. As
                            // such, no other thread can get it after it is
                            // destroyed. Since we put the guard into the
                            // vector, we are the ones responsible for getting
                            // rid of it.
                            unsafe { Value::defer_drop(desc, guard) };
                            // SAFETY: PushDescr is always a Value created from
                            // `Value::new_data`. We can still use `value`,
                            // because the descriptor didn't take it.
                            ph = unsafe { PushDescr::new_value(value.clone()) };
                        }
                    }
                    Result::Err(_) => {
                        // SAFETY: PushDescr is always a Value created from
                        // `Value::new_data`.
                        ph = unsafe { PushDescr::new_value(value.clone()) };
                    }
                }
            } else if is_descr(expected) {
                // SAFETY: We just checked there's a descriptor and expected
                // follows the tag convention.
                unsafe { Descriptor::complete_unchecked(expected, pos, &self) };
            } else {
                pos += 1;
            }
        }
    }

    /// Removes an element from the end of the vector.
    ///
    /// This corresponds to the `cas_popBack` method from the paper.
    pub fn pop(&self) -> Option<Ref<'_, T>> {
        let mut failures = 0;
        let mut ph = PopDescr::new();
        let mut pos = self.length.load(Ordering::SeqCst);
        let guard = &epoch::pin();
        loop {
            failures += 1;
            if failures > LIMIT {
                todo!("return announce_op(PopOp(value))")
            } else if pos == 0 {
                return Option::None;
            }
            // TODO: Use try_get_spot somehow to prevent over-allocating
            let spot = self.get_spot(pos, guard);
            let expected = spot.load(Ordering::SeqCst, guard);
            if expected.is_null() {
                let value_desc = Value::new_descriptor(Descriptor::Pop(ph));
                let value = Owned::new(value_desc).with_tag(1);
                match spot.compare_exchange(
                    expected,
                    value,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Result::Ok(desc) => {
                        // SAFETY: We know that there's a descriptor here and
                        // that it follows the tag convention, as we just put
                        // it there.
                        let res = unsafe { Descriptor::complete_unchecked(desc, pos, self) };
                        if res {
                            self.decrement_size();
                            // SAFETY: The descriptor's still there (if this
                            // causes UB, then the call to complete_unchecked
                            // must have caused UB too).
                            let value = unsafe { desc.deref().as_descriptor() };
                            let value = value.as_pop().unwrap();
                            // SAFETY: `value` was just completed, so calling
                            // `get_value` is safe.
                            return Option::Some(Ref::new(
                                self,
                                unsafe { value.get_value(guard) }.clone(),
                            ));
                        } else {
                            ph = PopDescr::new();
                            pos -= 1;
                        }
                    }
                    Result::Err(_) => {
                        ph = PopDescr::new();
                    }
                }
            }
        }
    }

    /// Gets the element at the specified position, or `None` if out-of-bounds.
    ///
    /// Note that out-of-bounds _may_ occur for an index less than
    /// `self.size()`, see [`Self::fast_push()`][a] for details.
    ///
    /// [a]: Self::fast_push()
    pub fn get(&self, index: usize) -> Option<Ref<'_, T>> {
        if index < self.length.load(Ordering::SeqCst) {
            let guard = &epoch::pin();
            let spot = self.get_spot(index, guard);
            let temp = spot.load(Ordering::SeqCst, guard);
            if is_descr(temp) {
                todo!("temp = temp.get_value(self, pos)")
            }
            if !temp.is_null() {
                // SAFETY: We know that `temp` is not null, and not a
                // descriptor, so it must be a value.
                return Option::Some(Ref::new(self, unsafe { temp.deref().as_data().clone() }));
            }
        }
        Option::None
    }

    /// Writes the value to the given index, returning the old value.
    ///
    /// Note that unlike [`Self::c_write`], this returns the _old_ value on
    /// success, while `c_write` returns the _new_ value. This is because with
    /// `c_write`, we know the old value, because it is equal to the value
    /// passed in to `old`.
    pub fn write(&self, index: usize, value: T) -> Result<Ref<'_, T>, T> {
        // SAFETY NOTE: `new` should always point to a valid `Data`. Until it
        // is placed into the vector, we own it, so we can safely read from it.
        let mut new = Owned::new(Value::new_data(value));
        let guard = &epoch::pin();
        let spot = self.try_get_spot(index, guard);
        for _failures in 0..LIMIT {
            let value = match &spot {
                Option::Some(spot) => spot.load(Ordering::SeqCst, guard),
                Option::None => Shared::null(),
            };
            if value.is_null() {
                // SAFETY: `new` always points to a valid `Data`. We own
                // it, so we can unwrap it from the `Arc`. (When we switch
                // to intrusive reference-counting, we can do this without
                // the check)
                return Result::Err(
                    Arc::try_unwrap(unsafe { new.into_box().into_data().val })
                        .unwrap_or_else(|_| unreachable!()),
                );
            } else if is_descr(value) {
                // SAFETY: This is a valid descriptor, as just checked, and
                // it follows the tag convention, so `is_descr` is a valid
                // check.
                unsafe { Descriptor::complete_unchecked(value, index, self) };
            } else {
                // If spot was not null, the first branch would have been
                // taken.
                match spot.unwrap().compare_exchange(
                    value,
                    new,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Result::Ok(_) => {
                        // SAFETY: `value` was taken from the vector and is
                        // neither null nor a descriptor, so therefore it
                        // must be a valid `Data`.
                        let value = unsafe { value.deref().as_data().clone() };
                        return Result::Ok(Ref::new(self, value));
                    }
                    Result::Err(err) => new = err.new,
                }
            }
        }
        todo!("return announce_op(WriteOp(index, new))")
    }

    /// Inserts the value at the given index.
    pub fn insert(&self, index: usize, value: T) {
        todo!()
    }

    /// Removes the value at the given index, shifting all elements after it
    /// to the left.
    pub fn remove(&self, index: usize) -> T {
        todo!()
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }

    /// Returns a reference to the `Atomic` at the given location, or `None` if
    /// the index is greater than the capacity.
    ///
    /// This differs from `get_spot` because it will not reallocate to extend
    /// in the case of an out-of-bounds index. If it is between `self.len()`
    /// and `self.capacity()`, the `Atomic` will be null, otherwise it will be
    /// a valid pointer that follows the tag convention.
    pub(crate) fn try_get_spot<'a>(
        &self,
        index: usize,
        guard: &'a Guard,
    ) -> Option<&'a Atomic<Value<T>>> {
        let data = self.data.load(Ordering::SeqCst, guard);
        if data.is_null() {
            Option::None
        } else {
            // SAFETY: If `self.data` is not null, it is safe to load.
            // FIXME: See `self.get_spot`.
            unsafe { data.deref() }.get(index).map(|x| &*x)
        }
    }

    /// Returns a reference to the `Atomic` at the given location.
    ///
    /// This will reallocate to ensure that the given index is in bounds if
    /// necessary. If the index is greater than `self.len()`, it will be null,
    /// otherwise it will be a valid pointer that follows the tag convention.
    pub(crate) fn get_spot<'a>(&self, index: usize, guard: &'a Guard) -> &'a Atomic<Value<T>> {
        loop {
            let data = self.data.load(Ordering::SeqCst, guard);
            if data.is_null() {
                match self.resize(4, data, guard) {
                    // SAFETY: We know that the data is initialized, and we can
                    // thus send a reference to it.
                    Result::Ok(x) => return unsafe { &x.deref()[index] },
                    Result::Err(_) => {}
                }
            } else {
                // SAFETY: If `self.data` is not null, it is safe to load.
                let slice = unsafe { data.deref() };
                match slice.get(index) {
                    // SAFETY: The reference won't outlive the guard, so it can't
                    // get destroyed accidentally. Furthermore, all data in
                    // `self.data` is initialized, so we can deref `as_ptr()`.
                    Option::Some(x) => return &x,
                    _ => {
                        match self.resize(slice.len().next_power_of_two(), data, guard) {
                            // SAFETY: We know that the data is initialized,
                            // and we can thus send a reference to it.
                            Result::Ok(x) => return unsafe { &x.deref()[index] },
                            Result::Err(_) => {}
                        }
                    }
                }
            }
        }
    }

    fn resize<'a>(
        &self,
        new_size: usize,
        self_data: Shared<PartialData<T>>,
        guard: &'a Guard,
    ) -> Result<Shared<'a, PartialData<T>>, ()> {
        if self_data.is_null() {
            let new_data = Owned::<PartialData<T>>::init(new_size);
            match self.data.compare_exchange(
                self_data,
                new_data,
                Ordering::SeqCst,
                Ordering::SeqCst,
                guard,
            ) {
                Result::Ok(x) => Result::Ok(x),
                Result::Err(_) => Result::Err(()),
            }
        } else {
            todo!()
        }
    }

    fn increment_size(&self) {
        self.length.fetch_add(1, Ordering::SeqCst);
    }

    fn decrement_size(&self) {
        self.length.fetch_sub(1, Ordering::SeqCst);
    }
}

impl<T: PartialEq> FvdVec<T> {
    /// Writes the given value to the given index, if not equal to the current
    /// value.
    ///
    /// If the value was equal, a [`Ref`] to the newly-placed value is
    /// returned. If not, the new value is returned intact.
    pub fn c_write(&self, index: usize, old: &T, new: T) -> Result<Ref<'_, T>, T> {
        // SAFETY NOTE: `new` should always point to a valid `Data`. Until it
        // is placed into the vector, we own it, so we can safely read from it.
        let mut new = Owned::new(Value::new_data(new));
        let guard = &epoch::pin();
        let spot = self.try_get_spot(index, guard);
        for _failures in 0..LIMIT {
            let value = match &spot {
                Option::Some(spot) => spot.load(Ordering::SeqCst, guard),
                Option::None => Shared::null(),
            };
            if value.is_null() {
                // SAFETY: `new` always points to a valid `Data`. We own
                // it, so we can unwrap it from the `Arc`. (When we switch
                // to intrusive reference-counting, we can do this without
                // the check)
                return Result::Err(
                    Arc::try_unwrap(unsafe { new.into_box().into_data().val })
                        .unwrap_or_else(|_| unreachable!()),
                );
            } else if is_descr(value) {
                // SAFETY: This is a valid descriptor, as just checked, and
                // it follows the tag convention, so `is_descr` is a valid
                // check.
                unsafe { Descriptor::complete_unchecked(value, index, self) };
            } else {
                // SAFETY: We know `value` is safe to dereference
                // because it came from `self` and is neither null nor a
                // descriptor; thus, it must point to valid data.
                let val = unsafe { value.deref().as_data() };
                if old == &**val {
                    // If spot was not null, the first branch would have
                    // been taken.
                    match spot.unwrap().compare_exchange(
                        value,
                        new,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                        guard,
                    ) {
                        Result::Ok(val) => {
                            // SAFETY: `val` comes from `new`, which still
                            // points to a valid `Data`.
                            let value = unsafe { val.deref().as_data().clone() };
                            return Result::Ok(Ref::new(self, value));
                        }
                        Result::Err(err) => new = err.new,
                    }
                } else {
                    // SAFETY: `new` always points to a valid `Data`. We
                    // own it, so we can unwrap it from the `Arc`. (When we
                    // switch to intrusive reference-counting, we can do
                    // this without the check)
                    return Result::Err(
                        Arc::try_unwrap(unsafe { new.into_box().into_data().val })
                            .unwrap_or_else(|_| unreachable!()),
                    );
                }
            }
        }
        todo!("return announce_op(CWriteOp(index, old, new))")
    }
}

impl<'a, T> Ref<'a, T> {
    fn new(parent: &'a FvdVec<T>, value: Node<T>) -> Ref<'a, T> {
        Ref {
            _parent: parent,
            value,
        }
    }
}

struct DebugHelper<'a, T>(Ref<'a, T>);

impl<T: Debug> Debug for DebugHelper<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <T as Debug>::fmt(&*self.0, f)
    }
}

impl<T: Debug> Debug for FvdVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.iter().map(DebugHelper))
            .finish()
    }
}

impl<T> Default for FvdVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq> PartialEq for FvdVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if ptr::eq(self, other) {
            // Fast path to prevent full iteration.
            // If this weren't here, it could be the case that comparing a
            // value with itself would return false b/c of the value being
            // changed out from under us while the comparison is happening.
            true
        } else if self.len() != other.len() {
            false
        } else {
            todo!()
        }
    }
}

impl<T: Eq + PartialEq> Eq for FvdVec<T> {}

impl<T> IntoIterator for FvdVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(mut self) -> Self::IntoIter {
        // SAFETY: We own the struct, so we can turn `self.data` into an
        // `Owned`. Furthermore, when we replace the data, we reset `self` to
        // be equivalent to an empty vector, so dropping it is still safe
        // (Since we implement `Drop`, we can't just move out). Furthermore,
        // just in case, we call `mem::forget` on `self`, to not accidentally
        // double-free. This won't leak because we now own no heap-allocated
        // memory anymore. Calling `IntoIter::from_parts` is safe because
        // `self.data` is always initialized up to `self.capacity`, which is
        // what we're passing as `data` and `end`, all values in `data` follow
        // the tag convention, and there are no initialized values beyond
        // `end`.
        unsafe {
            let data = mem::replace(&mut self.data, Atomic::null());
            let end = mem::replace(self.length.get_mut(), 0);
            mem::forget(self);
            IntoIter::from_parts(data.into_owned(), end)
        }
    }
}

impl<'a, T> IntoIterator for &'a FvdVec<T> {
    type Item = Ref<'a, T>;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, const N: usize> From<[T; N]> for FvdVec<T> {
    fn from(x: [T; N]) -> Self {
        let mut data = Owned::<PartialData<T>>::init(N);
        for (i, val) in array::IntoIter::new(x).enumerate() {
            data[i] = Atomic::new(Value::new_data(val));
        }
        FvdVec {
            data: Atomic::from(data),
            length: AtomicUsize::new(N),
            phantom: PhantomData,
        }
    }
}

impl<T> From<Box<[T]>> for FvdVec<T> {
    fn from(x: Box<[T]>) -> Self {
        Vec::from(x).into()
    }
}

impl<T> From<Vec<T>> for FvdVec<T> {
    fn from(x: Vec<T>) -> Self {
        let len = x.len();
        let mut data = Owned::<PartialData<T>>::init(len);
        for (i, val) in x.into_iter().enumerate() {
            data[i] = Atomic::new(Value::new_data(val));
        }
        FvdVec {
            data: Atomic::from(data),
            length: AtomicUsize::new(len),
            phantom: PhantomData,
        }
    }
}

struct FromIterHelper<T> {
    // SAFETY: All data in this up to `self.next_elem` is initialized with a
    // `Value` that has been created with `Value::new_data()`.
    data: Owned<PartialData<T>>,
    // SAFETY: `next_elem` always points to the first point in the slice that
    // is not properly initialized.
    next_elem: usize,
}

impl<T> FromIterHelper<T> {
    pub fn with_capacity(cap: usize) -> FromIterHelper<T> {
        FromIterHelper {
            data: Owned::init(cap),
            next_elem: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        if self.next_elem >= self.data.len() {
            let new_cap = (self.data.len() + 1).next_power_of_two();
            let mut new_data = Owned::<PartialData<T>>::init(new_cap);
            for (to, from) in new_data.iter_mut().zip(self.data.iter()) {
                // SAFETY: We're transferring data from the old value
                // (returning it to its previous, uninit state), and
                // writing it directly into the new value. Note that
                // whether or not `from` is initialized has no effect on
                // the safety of this code.
                // Furthermore, since this can't panic, there's no
                // chance of dropping uninitialized data.
                *to = unsafe { ptr::read(from) }
            }
            self.data = new_data;
        }
        // As promised, initialized properly.
        self.data[self.next_elem] = Atomic::new(Value::new_data(value));
        self.next_elem += 1;
    }

    pub fn take(mut self) -> (Owned<PartialData<T>>, usize) {
        // SAFETY: Not necessarily unsafe, but this does obey all the given
        // safety requirements of the type. The data is replaced with a slice
        // of length 0, and next_elem points at 0, so no uninitialized or
        // invalid memory is dropped.
        let len = mem::replace(&mut self.next_elem, 0);
        let data = mem::replace(&mut self.data, Owned::init(0));
        (data, len)
    }
}

impl<T> Drop for FromIterHelper<T> {
    fn drop(&mut self) {
        // Just in case something strange happens, set the length to 0 so we
        // don't cause UB by dropping something we shouldn't.
        let next_elem = mem::replace(&mut self.next_elem, 0);
        for val in &mut self.data[..next_elem] {
            // SAFETY: Every value up to `self.next_elem` is initialized
            // properly, as per the contract in the header. Furthermore, as we
            // are the only ones with access to this struct (see the
            // `&mut self`), we can turn it into an `Owned` to drop it.
            // Also, reading from `MaybeUninit` is safe, as long as we don't
            // call `assume_init` on it later (which we won't).
            unsafe {
                let data = ptr::read(val).into_owned();
                debug_assert_eq!(data.tag(), 0);
                let data = data.into_box().into_data();
                drop(data);
            }
        }
    }
}

impl<T> FromIterator<T> for FvdVec<T> {
    fn from_iter<U: IntoIterator<Item = T>>(iter: U) -> Self {
        let iterator = iter.into_iter();
        let size_hint = iterator.size_hint();
        let mut helper = FromIterHelper::with_capacity(size_hint.0);
        for val in iterator {
            helper.push(val)
        }
        let (data, length) = helper.take();
        FvdVec {
            data: Atomic::from(data),
            length: AtomicUsize::new(length),
            phantom: PhantomData,
        }
    }
}

impl<T> Drop for FvdVec<T> {
    fn drop(&mut self) {
        let data = mem::replace(&mut self.data, Atomic::null());
        // SAFETY: We mutably own this data structure, so there are no other
        // threads with a reference to it. As such, we really want to be able
        // to do this check on a value without using an atomic instruction,
        // but Crossbeam doesn't support that, so here we are.
        if data
            .load(Ordering::Relaxed, unsafe { epoch::unprotected() })
            .is_null()
        {
            return;
        }
        // SAFETY: We mutably own this data structure, so there are no other
        // threads containing a reference to it, nor are there any threads
        // still accessing any owned data. As such, we are allowed to create
        // an owned value from it. Furthermore, we have determined that it is
        // not null through the earlier check, and as we own it mutably, we are
        // sure nobody has modified it in the meantime.
        let length = *self.length.get_mut();
        for value in &unsafe { data.into_owned().deref() }[..length] {
            // SAFETY: Since we (still) own the data
            // structure, we can turn it into an Owned.
            let value = unsafe {
                let value = value.clone();
                if value
                    .load(Ordering::Relaxed, epoch::unprotected())
                    .is_null()
                {
                    continue;
                }
                value.into_owned()
            };
            // SAFETY: `value` follows the tag convention.
            // The `into_enum` call is necessary because otherwise we'd be
            // dropping a `MaybeUninit` (contained in the union), which does
            // not call `drop` on its contained value.
            unsafe {
                drop(Value::into_enum(value));
            }
        }
    }
}

impl<T> Deref for Ref<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value.val
    }
}

impl<T> Borrow<T> for Ref<'_, T> {
    fn borrow(&self) -> &T {
        &self.value.val
    }
}

#[cfg(test)]
mod tests {
    use crate::FvdVec;
    use std::array;

    #[test]
    fn test_new() {
        assert!(FvdVec::<()>::new().is_empty())
    }

    #[test]
    fn length() {
        assert_eq!(FvdVec::from([0, 1, 2, 3]).len(), 4)
    }

    #[test]
    fn capacity() {
        assert_eq!(FvdVec::<()>::with_capacity(10).capacity(), 10)
    }

    #[test]
    fn test_create() {
        assert_eq!(vec![0, 1, 2, 3].into_iter().collect::<FvdVec<_>>().len(), 4)
    }

    #[test]
    fn into_iter() {
        let values = [0, 1, 2, 3];
        let vec: FvdVec<_> = values.into();
        for (i, j) in vec.into_iter().zip(array::IntoIter::new(values)) {
            assert_eq!(i, j);
        }
    }

    #[test]
    fn push() {
        let vec = FvdVec::with_capacity(10);
        vec.push(0);
        assert_eq!(vec.len(), 1);
        for i in vec {
            assert_eq!(i, 0);
        }
    }

    #[test]
    fn pop() {
        let vec = FvdVec::with_capacity(10);
        vec.push(0);
        assert_eq!(*vec.pop().unwrap(), 0);
    }

    #[test]
    fn get() {
        let vec = FvdVec::from([0]);
        assert_eq!(*vec.get(0).unwrap(), 0)
    }

    #[test]
    fn write() {
        let vec = FvdVec::from([0]);
        vec.write(0, 1).unwrap();
        assert_eq!(*vec.get(0).unwrap(), 1);
    }

    #[test]
    fn c_write() {
        let vec = FvdVec::from([0]);
        assert_eq!(vec.c_write(0, &0, 1).map(|x| *x), Result::Ok(1));
        assert_eq!(vec.c_write(0, &0, 1).map(|x| *x), Result::Err(1));
    }

    #[test]
    fn empty_resize() {
        let vec = FvdVec::new();
        vec.push(0);
        assert_eq!(*vec.get(0).unwrap(), 0);
    }
}
