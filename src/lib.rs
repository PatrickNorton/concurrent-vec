//! A wait-free vector based on a [paper][a] by Feldman, Valera-Leon, and
//! Dechev.
//!
//! It supports random-access reads and writes, push/pop, and
//! insert/remove operations.
//!
//! [a]: https://www.osti.gov/servlets/purl/1427291

use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;

/// A wait-free vector based on the paper by Feldman et al. It offers
/// random-access reads and writes, as well as push/pop operations.
pub struct FvdVec<T> {
    phantom: PhantomData<T>,
}

impl<T> FvdVec<T> {
    /// Creates a new, empty `WFVec<T>`.
    ///
    /// Until an implementation is provided, this method cannot be `const`.
    /// (panicking in const fn is still unstable, see [#51999][a])
    ///
    /// [a]: https://github.com/rust-lang/rust/issues/51999
    pub fn new() -> FvdVec<T> {
        todo!()
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
        todo!()
    }

    /// Checks if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the current capacity of the vector.
    pub fn capacity(&self) -> usize {
        todo!()
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
        todo!()
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
