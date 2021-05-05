//! A wait-free vector based on a [paper][a] by Feldman, Valera-Leon, and
//! Dechev.
//!
//! It supports random-access reads and writes, push/pop, and
//! insert/remove operations.
//!
//! [a]: https://www.osti.gov/servlets/purl/1427291

mod descr;
mod iter;

use crate::descr::{is_descr, Descriptor, Node, PushDescr, Value};
use crate::iter::{IntoIter, Iter};
use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};
use std::borrow::Borrow;
use std::fmt::{self, Debug, Formatter};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{array, mem, ptr};

type Data<T> = [MaybeUninit<Atomic<Value<T>>>];

const LIMIT: usize = 32;

/// A wait-free vector based on the paper by Feldman et al. It offers
/// random-access reads and writes, as well as push/pop operations.
pub struct FvdVec<T> {
    // SAFETY GUARANTEES:
    // * self.data is either null or points to a valid slice.
    // * * If self.data is null, this vector is empty.
    // * All values in the array are initialized, though maybe with null.
    // * All `Atomic<Value<T>>` that are non-null and non-`NOT_VALUE` follow
    //   the tag convention.
    data: Atomic<Data<T>>,
    length: AtomicUsize,
    capacity: AtomicUsize,
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
            capacity: AtomicUsize::new(0),
            phantom: PhantomData,
        }
    }

    /// Creates a `WFVec<T>` with the given capacity.
    pub fn with_capacity(cap: usize) -> FvdVec<T> {
        let mut data = Owned::<Data<T>>::init(cap);
        // FIXME: Buffer overflow in crossbeam (array initialized with length
        //  equal to total allocated bytes, not number of elements)
        data[..cap].fill_with(|| MaybeUninit::new(Atomic::null()));
        FvdVec {
            data: Atomic::from(data),
            length: AtomicUsize::new(0),
            capacity: AtomicUsize::new(cap),
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
        self.capacity.load(Ordering::SeqCst)
    }

    /// Reserves space for `additional` values to be inserted in `self`.
    pub fn reserve(&self, additional: usize) {
        todo!()
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
    pub fn pop(&self) -> Option<T> {
        todo!()
    }

    /// Appends an element to the end of the vector.
    ///
    /// This corresponds to the `faa_pushBack` method from the paper.
    ///
    /// # Interaction with indexing
    ///
    /// Use of this method in conjunction with indexing can result in
    /// unexpected behavior. If this method has begun, but not fully completed,
    /// the index it references is still considered out-of-bounds until the
    /// value is written to it. As such, it is possible for an index to a
    /// value less than `self.len()` to fail.
    pub fn fast_push(&self, value: T) {
        todo!()
    }

    /// Removes an element from the end of the vector.
    ///
    /// This corresponds to the `faa_popBack` method from the paper.
    pub fn fast_pop(&self) -> Option<T> {
        todo!()
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
                return Option::Some(Ref {
                    _parent: self,
                    value: unsafe { temp.deref().as_data().clone() },
                });
            }
        }
        Option::None
    }

    /// Writes the given value to the given index.
    ///
    /// This will return false if `index` is out-of-bounds, or if another
    /// thread is in the middle of updating it (I think)
    ///
    /// TODO: Get better explanation
    pub fn c_write(&self, index: usize, value: T) -> Result<(), T> {
        todo!()
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

    pub(crate) fn get_spot<'a>(&self, index: usize, guard: &'a Guard) -> &'a Atomic<Value<T>> {
        let data = self.data.load(Ordering::SeqCst, guard);
        if data.is_null() {
            todo!("Resize")
        } else {
            // SAFETY: If `self.data` is not null, it is safe to load.
            match unsafe { data.deref() }.get(index) {
                // SAFETY: The reference won't outlive the guard, so it can't
                // get destroyed accidentally. Furthermore, all data in
                // `self.data` is initialized, so we can deref `as_ptr()`.
                Option::Some(x) => unsafe { &*x.as_ptr() },
                Option::None => todo!("Resize"),
            }
        }
    }

    fn resize<'a>(
        &self,
        new_size: usize,
        self_data: Shared<Data<T>>,
        guard: &'a Guard,
    ) -> Result<Shared<'a, Data<T>>, ()> {
        todo!()
    }

    fn increment_size(&self) {
        self.length.fetch_add(1, Ordering::SeqCst);
    }
}

impl<T: Debug> Debug for FvdVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        todo!()
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
            *self.capacity.get_mut() = 0;
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
        let mut data = Owned::<Data<T>>::init(N);
        for (i, val) in array::IntoIter::new(x).enumerate() {
            data[i] = MaybeUninit::new(Atomic::new(Value::new_data(val)));
        }
        FvdVec {
            data: Atomic::from(data),
            length: AtomicUsize::new(N),
            capacity: AtomicUsize::new(N),
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
        let mut data = Owned::<Data<T>>::init(len);
        for (i, val) in x.into_iter().enumerate() {
            data[i] = MaybeUninit::new(Atomic::new(Value::new_data(val)));
        }
        FvdVec {
            data: Atomic::from(data),
            length: AtomicUsize::new(len),
            capacity: AtomicUsize::new(len),
            phantom: PhantomData,
        }
    }
}

struct FromIterHelper<T> {
    // SAFETY: All data in this up to `self.next_elem` is initialized with a
    // `Value` that has been created with `Value::new_data()`.
    data: Owned<Data<T>>,
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
            let mut new_data = Owned::<Data<T>>::init(new_cap);
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
        self.data[self.next_elem] = MaybeUninit::new(Atomic::new(Value::new_data(value)));
        self.next_elem += 1;
    }

    pub fn take(mut self) -> (Owned<Data<T>>, usize) {
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
                let data = ptr::read(val).assume_init().into_owned();
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
        let capacity = helper.data.len();
        let (data, length) = helper.take();
        FvdVec {
            data: Atomic::from(data),
            length: AtomicUsize::new(length),
            capacity: AtomicUsize::new(capacity),
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
            // SAFETY: Because we can't turn this slice into a box, as it is
            // unsized, we have to read it from the slice before we can assume
            // anything. Furthermore, it is guaranteed that all values up to
            // self.length are initialized (if only with null), so calling
            // assume_init is safe. Furthermore, since we (still) own the data
            // structure, we can turn it into an Owned.
            let value = unsafe {
                let value = ptr::read(value).assume_init();
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
    fn get() {
        let vec = FvdVec::from([0]);
        assert_eq!(*vec.get(0).unwrap(), 0)
    }
}
