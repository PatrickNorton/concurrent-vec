//! A wait-free vector based on a [paper][a] by Feldman, Valera-Leon, and
//! Dechev.
//!
//! It supports random-access reads and writes, push/pop, and
//! insert/remove operations.
//!
//! [a]: https://www.osti.gov/servlets/purl/1427291

mod descr;
mod iter;

use crate::descr::{Node, Value};
use crate::iter::IntoIter;
use crossbeam::epoch::{self, Atomic, Owned};
use std::fmt::{self, Debug, Formatter};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{array, mem, ptr};

/// A wait-free vector based on the paper by Feldman et al. It offers
/// random-access reads and writes, as well as push/pop operations.
pub struct FvdVec<T> {
    // SAFETY GUARANTEES:
    // * self.data is either null or points to a valid slice.
    // * * If self.data is null, this vector is empty.
    // * All values up to self.length are initialized, though maybe with null.
    // * * We may change this so that array resizing initializes all values
    //     with Atomic::null(), so that this isn't an issue.
    data: Atomic<[MaybeUninit<Atomic<Value<T>>>]>,
    length: AtomicUsize,
    capacity: AtomicUsize,
    phantom: PhantomData<T>,
}

pub struct Ref<'a, T> {
    parent: &'a FvdVec<T>,
    value: &'a Node<T>,
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
        FvdVec {
            data: Atomic::init(cap),
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
    ///
    /// This corresponds to the `cas_pushBack` method from the paper.
    pub fn push(&self, value: T) {
        todo!()
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
    pub fn get(&self, index: usize) -> Option<&T> {
        todo!()
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

    pub(crate) fn get_spot(&self, index: usize) -> &Atomic<Value<T>> {
        todo!()
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
        todo!()
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
            IntoIter::from_parts(data.into_owned(), end)
        }
    }
}

impl<T, const N: usize> From<[T; N]> for FvdVec<T> {
    fn from(x: [T; N]) -> Self {
        let mut data = Owned::<[MaybeUninit<Atomic<Value<T>>>]>::init(N);
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
        let mut data = Owned::<[MaybeUninit<Atomic<Value<T>>>]>::init(len);
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
    data: Owned<[MaybeUninit<Atomic<Value<T>>>]>,
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
            let mut new_data = Owned::<[MaybeUninit<Atomic<Value<T>>>]>::init(new_cap);
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

    pub fn take(mut self) -> (Owned<[MaybeUninit<Atomic<Value<T>>>]>, usize) {
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
}
