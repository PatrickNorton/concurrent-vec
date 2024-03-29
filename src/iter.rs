use crate::data::PartialData;
use crate::descr::{Value, ValueEnum};
use crate::{FvdVec, Ref};
use crossbeam_epoch::{self as epoch, Owned};
use std::iter::FusedIterator;
use std::mem::take;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub struct IntoIter<T> {
    data: Owned<PartialData<T>>,
    next: usize,
    end: usize,
}

pub struct Iter<'a, T> {
    data: &'a FvdVec<T>,
    next: usize,
}

impl<T> IntoIter<T> {
    /// Creates an IntoIter<T> from its component parts.
    ///
    /// SAFETY:
    /// * `data` must be initialized from 0 to `end`.
    /// * All `Atomic`s in `data` must follow the tag convention.
    /// * Any initialized values in slots `end..` will not be dropped.
    pub unsafe fn from_parts(data: Owned<PartialData<T>>, end: usize) -> IntoIter<T> {
        IntoIter { data, next: 0, end }
    }
}

impl<'a, T> Iter<'a, T> {
    pub fn new(data: &'a FvdVec<T>) -> Iter<'a, T> {
        Iter { data, next: 0 }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.next >= self.end {
                return Option::None;
            } else {
                // SAFETY: This is safe because:
                // * All values less that `self.end` are initialized.
                // * We own the struct, so we can turn the values into `Owned`.
                // * The `Owned` we get follows the tag convention for values,
                //   so turning it into an enum is safe.
                unsafe {
                    let atomic = take(&mut self.data[self.next]);
                    // Skip over null values. There should be an easier way to
                    // do this, but there isn't. We can use `unprotected`
                    // because we have a mutable reference, so it won't be
                    // dropped. This branch probably shouldn't ever be taken,
                    // as we have a mutable reference to self, so all
                    // operations should be finished, but I'm not 100% sure,
                    // and this way there's no UB.
                    if atomic
                        .load(Ordering::Relaxed, epoch::unprotected())
                        .is_null()
                    {
                        continue;
                    }
                    self.next += 1;
                    match Value::into_enum(atomic.into_owned()) {
                        // TODO: Intrusive reference counting should remove
                        //  this `unwrap`.
                        ValueEnum::Data(d) => {
                            return Option::Some(
                                Arc::try_unwrap(d.val).unwrap_or_else(|_| panic!()),
                            );
                        }
                        // In theory, these shouldn't exist (all descriptor-
                        // based operations should have finished well before
                        // a mutable reference is created), but we choose to
                        // deal with them here anyways, as a form of
                        // redundancy (causing UB is bad, and this doesn't cost
                        // much).
                        ValueEnum::Push(_) => {}
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.next;
        (remaining, Option::Some(remaining))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> FusedIterator for IntoIter<T> {}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = Ref<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Super simple implementation that works. Note that because of the
        // fact that the underlying data can be concurrently modified, it is
        // not a `FusedIterator`. A more complicated implementation could make
        // this Fused, but would need to take heed of the fact that `fast_push`
        // and `fast_pop` can create intermediary empty values.
        let value = self.data.get(self.next);
        self.next += 1;
        value
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // We have to clean out the remaining items, or they won't get dropped .
        // at all. There might be slightly faster ways to do it (in particular
        // using `ptr::drop_in_place`), but this works well enough and is
        // simple.
        for value in self {
            drop(value)
        }
    }
}
