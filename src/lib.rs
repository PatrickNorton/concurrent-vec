//! A wait-free vector based on a [paper][a] by Feldman, Valera-Leon, and
//! Dechev.
//!
//! It supports random-access reads and writes, push/pop, and
//! insert/remove operations.
//!
//! [a]: https://www.osti.gov/servlets/purl/1427291

mod descr;

use crate::descr::{Node, Value};
use crossbeam::epoch::{self, Atomic};
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{mem, ptr};

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
        todo!()
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
            // Owned calls the destructor, so we're good here :)
            drop(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::FvdVec;

    #[test]
    fn it_works() {
        assert!(FvdVec::<()>::new().is_empty())
    }
}
