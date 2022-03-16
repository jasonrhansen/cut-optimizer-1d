//! cut-optimizer-1d is an optimizer library that attempts to figure out an optimal cutting of
//! linear cut lengths from linear stock lengths with the least amount of waste.
//! It uses genetic algorithms and multiple heuristics to solve the problem.

#![deny(missing_docs)]

mod basic;
mod genetic;

#[cfg(test)]
mod tests;

use basic::BasicBin;
use fnv::FnvHashSet;
use genetic::population::Population;
use genetic::unit::Unit;
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::borrow::Borrow;
use std::cmp;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// A rectangular piece that needs to be cut from a stock piece.
#[cfg_attr(feature = "serialize", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serialize", serde(rename_all = "camelCase"))]
#[derive(Clone, Debug)]
pub struct CutPiece {
    /// Quantity of this cut piece.
    pub quantity: usize,

    /// ID to be used by the caller to match up result cut pieces
    /// with the original cut piece. This ID has no meaning to the
    /// optimizer so it can be set to `None` if not needed.
    pub external_id: Option<usize>,

    /// Length of this cut piece.
    pub length: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct CutPieceWithId {
    pub(crate) id: usize,
    pub(crate) external_id: Option<usize>,
    pub(crate) length: usize,
}

impl Hash for CutPieceWithId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}
impl PartialEq for CutPieceWithId {
    fn eq(&self, other: &CutPieceWithId) -> bool {
        self.id == other.id
    }
}
impl Eq for CutPieceWithId {}

#[derive(Clone, Debug)]
pub(crate) struct UsedCutPiece {
    pub(crate) id: usize,
    pub(crate) external_id: Option<usize>,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

impl UsedCutPiece {
    pub(crate) fn length(&self) -> usize {
        self.end - self.start
    }
}

impl PartialEq for UsedCutPiece {
    fn eq(&self, other: &UsedCutPiece) -> bool {
        self.id == other.id
    }
}
impl Eq for UsedCutPiece {}

impl From<&UsedCutPiece> for CutPieceWithId {
    fn from(used_cut_piece: &UsedCutPiece) -> Self {
        Self {
            id: used_cut_piece.id,
            external_id: used_cut_piece.external_id,
            length: used_cut_piece.end - used_cut_piece.start,
        }
    }
}

impl From<&UsedCutPiece> for ResultCutPiece {
    fn from(used_cut_piece: &UsedCutPiece) -> Self {
        Self {
            external_id: used_cut_piece.external_id,
            start: used_cut_piece.start,
            end: used_cut_piece.end,
        }
    }
}

/// A cut piece that has been placed in a solution by the optimizer.
#[cfg_attr(feature = "serialize", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serialize", serde(rename_all = "camelCase"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultCutPiece {
    /// ID that matches the one on the cut piece that was passed to the optimizer.
    pub external_id: Option<usize>,

    /// Start location of this cut piece within the stock piece.
    pub start: usize,

    /// End location of this cut piece within the stock piece.
    pub end: usize,
}

/// A rectangular stock piece that is available to cut one or more
/// cut pieces from.
#[cfg_attr(feature = "serialize", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serialize", serde(rename_all = "camelCase"))]
#[derive(Hash, Copy, Clone, Debug, Eq, PartialEq)]
pub struct StockPiece {
    /// Length of rectangular stock piece.
    pub length: usize,

    /// Price to use to optimize for price when not all stock pieces are the same price per unit
    /// length. If optimizing for less waste instead, price can be set to 0 for all stock pieces.
    pub price: usize,

    /// Quantity of this stock piece available for optimization. `None` means infinite quantity.
    pub quantity: Option<usize>,
}

impl StockPiece {
    /// Decrement the quantity of this stock piece. If quantity is `None` it will remain `None`.
    fn dec_quantity(&mut self) {
        if let Some(ref mut quantity) = self.quantity {
            *quantity -= 1;
        }
    }

    /// Increment the quantity of this stock piece. If quantity is `None` it will remain `None`.
    fn inc_quantity(&mut self) {
        if let Some(ref mut quantity) = self.quantity {
            *quantity += 1;
        }
    }
}

/// Stock piece that was used by the optimizer to get one or more cut pieces.
#[cfg_attr(feature = "serialize", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serialize", serde(rename_all = "camelCase"))]
#[derive(Clone, Debug)]
pub struct ResultStockPiece {
    /// Length of this stock piece.
    pub length: usize,

    /// Cut pieces to cut from this stock piece.
    pub cut_pieces: Vec<ResultCutPiece>,

    /// Price of stock piece.
    pub price: usize,
}

/// Represents a bin used for bin-packing.
trait Bin {
    /// Creates a new `Bin`.
    fn new(length: usize, blade_width: usize, price: usize) -> Self;

    /// Computes the fitness of this `Bin` on a scale of 0.0 to 1.0, with 1.0 being the most fit.
    fn fitness(&self) -> f64;

    fn price(&self) -> usize;

    /// Removes `UsedCutPiece`s from this `Bin` and returns how many were removed.
    fn remove_cut_pieces<I>(&mut self, cut_pieces: I) -> usize
    where
        I: Iterator,
        I::Item: Borrow<UsedCutPiece>;

    /// Returns an iterator over the `UsedCutPiece`s in this `Bin`.
    fn cut_pieces(&self) -> std::slice::Iter<'_, UsedCutPiece>;

    /// Inserts the `CutPieceWithId` into this `Bin`. Returns whether the insert succeeded.
    fn insert_cut_piece(&mut self, cut_piece: &CutPieceWithId) -> bool;

    /// Returns whether the `StockPiece` is equivalent to this `Bin`.
    fn matches_stock_piece(&self, stock_piece: &StockPiece) -> bool;
}

struct OptimizerUnit<'a, B>
where
    B: Bin,
{
    bins: Vec<B>,

    // All of the possible stock pieces. It remains constant.
    possible_stock_pieces: &'a [StockPiece],

    // Stock pieces that are currently available to use for new bins.
    available_stock_pieces: Vec<StockPiece>,

    // Cut pieces that couldn't be added to bins.
    unused_cut_pieces: HashSet<CutPieceWithId>,

    blade_width: usize,
}

impl<'a, B> Clone for OptimizerUnit<'a, B>
where
    B: Bin + Clone,
{
    fn clone(&self) -> Self {
        Self {
            bins: self.bins.clone(),
            possible_stock_pieces: self.possible_stock_pieces,
            available_stock_pieces: self.available_stock_pieces.to_vec(),
            unused_cut_pieces: self.unused_cut_pieces.clone(),
            blade_width: self.blade_width,
        }
    }
}

impl<'a, B> OptimizerUnit<'a, B>
where
    B: Bin,
{
    fn new<R>(
        possible_stock_pieces: &'a [StockPiece],
        cut_pieces: &[&CutPieceWithId],
        blade_width: usize,
        rng: &mut R,
    ) -> Result<OptimizerUnit<'a, B>>
    where
        R: Rng + ?Sized,
    {
        let mut unit = OptimizerUnit {
            bins: Vec::new(),
            possible_stock_pieces,
            available_stock_pieces: possible_stock_pieces.to_vec(),
            unused_cut_pieces: Default::default(),
            blade_width,
        };

        for cut_piece in cut_pieces {
            if !unit.insert_cut_piece(cut_piece, rng) {
                unit.unused_cut_pieces.insert((*cut_piece).clone());
            }
        }

        Ok(unit)
    }

    pub(crate) fn generate_initial_units(
        possible_stock_pieces: &'a [StockPiece],
        mut cut_pieces: Vec<&CutPieceWithId>,
        blade_width: usize,
        random_seed: u64,
    ) -> Result<Vec<OptimizerUnit<'a, B>>> {
        let length_set: FnvHashSet<usize> = cut_pieces.iter().map(|p| p.length).collect();
        let unique_cut_lengths = length_set.len();

        let num_units = {
            let denom = if cut_pieces.len() > 1 {
                (cut_pieces.len() as f64).log10()
            } else {
                1.0
            };

            cmp::max(
                10,
                (cut_pieces.len() as f64 / denom + ((unique_cut_lengths - 1) * 10) as f64) as usize,
            )
        };
        let mut units = Vec::with_capacity(num_units);
        let mut rng: StdRng = SeedableRng::seed_from_u64(random_seed);

        // Sorted by length ascending
        cut_pieces.sort_by_key(|p| p.length);
        units.push(OptimizerUnit::new(
            possible_stock_pieces,
            &cut_pieces,
            blade_width,
            &mut rng,
        )?);

        // Sorted by length descending
        cut_pieces.sort_by_key(|p| cmp::Reverse(p.length));
        units.push(OptimizerUnit::new(
            possible_stock_pieces,
            &cut_pieces,
            blade_width,
            &mut rng,
        )?);

        // The rest randomly ordered
        for _ in 0..num_units - units.len() {
            cut_pieces.shuffle(&mut rng);
            units.push(OptimizerUnit::new(
                possible_stock_pieces,
                &cut_pieces,
                blade_width,
                &mut rng,
            )?);
        }

        Ok(units)
    }

    fn insert_cut_piece<R>(&mut self, cut_piece: &CutPieceWithId, rng: &mut R) -> bool
    where
        R: Rng + ?Sized,
    {
        for bin in self.bins.iter_mut() {
            if bin.insert_cut_piece(cut_piece) {
                return true;
            }
        }

        self.add_to_new_bin(cut_piece, rng)
    }

    fn add_to_new_bin<R>(&mut self, cut_piece: &CutPieceWithId, rng: &mut R) -> bool
    where
        R: Rng + ?Sized,
    {
        let stock_pieces = self
            .available_stock_pieces
            .iter_mut()
            .filter(|stock_piece| {
                stock_piece.quantity != Some(0) && cut_piece.length <= stock_piece.length
            });

        match stock_pieces.choose(rng) {
            Some(stock_piece) => {
                stock_piece.dec_quantity();

                let mut bin = B::new(stock_piece.length, self.blade_width, stock_piece.price);
                if !bin.insert_cut_piece(cut_piece) {
                    return false;
                }
                self.bins.push(bin);
                true
            }
            None => false,
        }
    }

    fn crossover<R>(&self, other: &OptimizerUnit<'a, B>, rng: &mut R) -> OptimizerUnit<'a, B>
    where
        R: Rng + ?Sized,
        B: Clone,
    {
        // If there aren't multiple bins we can't do a crossover,
        // so just return a clone of this unit.
        if self.bins.len() < 2 && other.bins.len() < 2 {
            return self.clone();
        }

        // Randomly select insertion point and range of bins to inject into the new unit.
        let cross_dest = rng.gen_range(0..=self.bins.len());
        let cross_src_start = rng.gen_range(0..other.bins.len());
        let cross_src_end = rng.gen_range(cross_src_start + 1..=other.bins.len());

        // Create a new unit with the bins of `self` and some of the bins of `other` injected in.
        let mut new_unit = OptimizerUnit {
            bins: (&self.bins[..cross_dest])
                .iter()
                .chain((&other.bins[cross_src_start..cross_src_end]).iter())
                .chain((&self.bins[cross_dest..]).iter())
                .cloned()
                .collect(),
            possible_stock_pieces: self.possible_stock_pieces,
            // Make all possible stock pieces available initially. Quantities will be updated below.
            available_stock_pieces: self.possible_stock_pieces.to_vec(),
            // Start with no unused cut pieces, and update below.
            unused_cut_pieces: Default::default(),
            blade_width: self.blade_width,
        };

        let mut unused_cut_pieces = self.unused_cut_pieces.clone();

        // Update available stock piece quantities based on the injected bins.
        other.bins[cross_src_start..cross_src_end]
            .iter()
            .for_each(|bin| {
                if let Some(ref mut stock_piece) = new_unit
                    .available_stock_pieces
                    .iter_mut()
                    .find(|sp| bin.matches_stock_piece(sp))
                {
                    // Remove the injected cut pieces from the unused set.
                    for cut_piece in bin.cut_pieces() {
                        unused_cut_pieces.remove(&cut_piece.into());
                    }
                    stock_piece.dec_quantity();
                } else {
                    panic!("Attempt to inject invalid bin in crossover operation. This shouldn't happen, and means there is a bug in the code.");
                }
            });

        // Remove injected cut pieces from all non-injected bins,
        // and update the available stock piece quantities.
        for i in (0..cross_dest)
            .chain((cross_dest + cross_src_end - cross_src_start)..new_unit.bins.len())
            .rev()
        {
            let bin = &mut new_unit.bins[i];
            let stock_piece = new_unit
                .available_stock_pieces
                .iter_mut()
                .find(|sp| sp.quantity != Some(0) && bin.matches_stock_piece(sp));
            let injected_cut_pieces = (&other.bins[cross_src_start..cross_src_end])
                .iter()
                .flat_map(Bin::cut_pieces);
            let num_removed_cut_pieces = bin.remove_cut_pieces(injected_cut_pieces);

            if let (0, Some(stock_piece)) = (num_removed_cut_pieces, stock_piece) {
                // No cut pieces were removed from this bin, so we'll keep it
                // and decrement the quantity of the available stock piece.
                stock_piece.dec_quantity();
            } else {
                // Either there were no available stock pieces, or there
                // were cut pieces removed from this bin so we'll remove it
                // and add its cut pieces to the unused set.
                for cut_piece in bin.cut_pieces() {
                    unused_cut_pieces.insert(cut_piece.into());
                }
                new_unit.bins.remove(i);
            }
        }

        // Try to add all unused cut pieces.
        unused_cut_pieces.retain(|cut_piece| !new_unit.insert_cut_piece(cut_piece, rng));
        new_unit.unused_cut_pieces = unused_cut_pieces;

        // Remove bins that don't have cut pieces.
        for i in (0..new_unit.bins.len()).rev() {
            if new_unit.bins[i].cut_pieces().next().is_none() {
                let bin = &mut new_unit.bins[i];
                if let Some(ref mut stock_piece) = new_unit
                    .available_stock_pieces
                    .iter_mut()
                    .find(|sp| sp.quantity != Some(0) && bin.matches_stock_piece(sp))
                {
                    // Make this stock piece available to use again.
                    stock_piece.inc_quantity();
                }
                new_unit.bins.remove(i);
            }
        }

        new_unit
    }

    // Randomly apply a mutation to this unit.
    fn mutate<R>(&mut self, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        if !self.bins.is_empty() && rng.gen_range(0..20) == 1 {
            self.inversion(rng)
        }
    }

    // Reverse order of a random range of bins.
    fn inversion<R>(&mut self, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        let start = rng.gen_range(0..self.bins.len());
        let end = rng.gen_range(start..self.bins.len());
        self.bins[start..end].reverse();
    }
}

impl<'a, B> Unit for OptimizerUnit<'a, B>
where
    B: Bin + Send + Clone,
{
    fn fitness(&self) -> f64 {
        let fitness = if self.bins.is_empty() {
            0.0
        } else {
            self.bins.iter().fold(0.0, |acc, b| acc + b.fitness()) / self.bins.len() as f64
        };

        if self.unused_cut_pieces.is_empty() {
            fitness
        } else {
            // If there are unused cut pieces, the fitness is below 0 because it's not a valid
            // solution.
            fitness - 1.0
        }
    }

    fn breed_with<R>(&self, other: &OptimizerUnit<'a, B>, rng: &mut R) -> OptimizerUnit<'a, B>
    where
        R: Rng + ?Sized,
    {
        let mut new_unit = self.crossover(other, rng);
        new_unit.mutate(rng);
        new_unit
    }
}

/// Error while optimizing.
#[derive(Debug)]
pub enum Error {
    /// There was no stock piece that could contain this demand piece.
    NoFitForCutPiece(CutPiece),
}
fn no_fit_for_cut_piece_error(cut_piece: &CutPieceWithId) -> Error {
    Error::NoFitForCutPiece(CutPiece {
        quantity: 1,
        external_id: cut_piece.external_id,
        length: cut_piece.length,
    })
}
type Result<T> = std::result::Result<T, Error>;

/// A valid solution to an optimization.
#[cfg_attr(feature = "serialize", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serialize", serde(rename_all = "camelCase"))]
pub struct Solution {
    /// Fitness score for this solution.
    /// Ranges between 0.0 and 1.0 inclusive, with 1.0 being a perfect solution with no waste.
    pub fitness: f64,

    /// The stock pieces that were used for this solution, each containing the demand piece layout.
    pub stock_pieces: Vec<ResultStockPiece>,

    #[cfg_attr(feature = "serialize", serde(skip))]
    price: usize,
}

/// Optimizer for optimizing rectangular cut pieces from rectangular
/// stock pieces.
pub struct Optimizer {
    stock_pieces: Vec<StockPiece>,
    cut_pieces: Vec<CutPieceWithId>,
    cut_width: usize,
    random_seed: u64,
    allow_mixed_stock_sizes: bool,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self {
            stock_pieces: Default::default(),
            cut_pieces: Default::default(),
            cut_width: Default::default(),
            random_seed: Default::default(),
            allow_mixed_stock_sizes: true,
        }
    }
}

impl Optimizer {
    /// Create a new optimizer.
    pub fn new() -> Self {
        Default::default()
    }

    /// Add a stock piece that the optimizer can use to optimize cut pieces.
    /// If the same stock piece is added multiple times, the quantities will be
    /// summed up. If any have a `None` quantity, the quantity on other equivalent
    /// pieces will be ignored.
    pub fn add_stock_piece(&mut self, stock_piece: StockPiece) -> &mut Self {
        let mut existing_stock_piece = self
            .stock_pieces
            .iter_mut()
            .find(|sp| sp.length == stock_piece.length && sp.price == stock_piece.price);

        if let Some(ref mut existing_stock_piece) = existing_stock_piece {
            match (&mut existing_stock_piece.quantity, stock_piece.quantity) {
                (Some(ref mut existing_quantity), Some(quantity)) => {
                    // If there is already a stock piece that is the same except the quantity, add
                    // to the quantity.
                    *existing_quantity += quantity;
                }
                _ => {
                    // `None` quantity means infinite, so if either is `None` we want it to be
                    // `None`.
                    existing_stock_piece.quantity = None;
                }
            }
        } else {
            // A stock piece like this hasn't yet been added so let's do it.
            self.stock_pieces.push(stock_piece);
        }

        self
    }

    /// Add a stock pieces that the optimizer can use to optimize cut pieces.
    /// If the same stock piece is added multiple times, the quantities will be
    /// summed up. If any have a `None` quantity, the quantity on other equivalent
    /// pieces will be ignored.
    pub fn add_stock_pieces<I>(&mut self, stock_pieces: I) -> &mut Self
    where
        I: IntoIterator<Item = StockPiece>,
    {
        stock_pieces.into_iter().for_each(|sp| {
            self.add_stock_piece(sp);
        });
        self
    }

    /// Add a desired cut piece that you need cut from a stock piece.
    pub fn add_cut_piece(&mut self, cut_piece: CutPiece) -> &mut Self {
        for _ in 0..cut_piece.quantity {
            let cut_piece = CutPieceWithId {
                id: self.cut_pieces.len(),
                external_id: cut_piece.external_id,
                length: cut_piece.length,
            };

            self.cut_pieces.push(cut_piece);
        }

        self
    }

    /// Add desired cut pieces that you need cut from a stock piece.
    pub fn add_cut_pieces<I>(&mut self, cut_pieces: I) -> &mut Self
    where
        I: IntoIterator<Item = CutPiece>,
    {
        cut_pieces.into_iter().for_each(|dp| {
            self.add_cut_piece(dp);
        });
        self
    }

    /// Set the width of the cut to use between cut pieces. This could
    /// represent blade or kerf thickness.
    pub fn set_cut_width(&mut self, cut_width: usize) -> &mut Self {
        self.cut_width = cut_width;
        self
    }

    /// Set the random seed used by the genetic algorithms in the optimizer. Using
    /// the same random seed will give you the same result for the same input.
    pub fn set_random_seed(&mut self, seed: u64) -> &mut Self {
        self.random_seed = seed;
        self
    }

    /// Set whether the optimizer should allow mixed sized stock pieces in the results.
    /// If set to false, and multiple stock sizes are given, only one stock size will be used in
    /// the results.
    pub fn allow_mixed_stock_sizes(&mut self, allow: bool) -> &mut Self {
        self.allow_mixed_stock_sizes = allow;
        self
    }

    /// Perform optimization
    pub fn optimize<F>(&self, progress_callback: F) -> Result<Solution>
    where
        F: Fn(f64),
    {
        // If there are no cut pieces, there's nothing to optimize.
        if self.cut_pieces.is_empty() {
            return Ok(Solution {
                fitness: 1.0,
                stock_pieces: Vec::new(),
                price: 0,
            });
        }

        let size_set: FnvHashSet<usize> = self.stock_pieces.iter().map(|sp| sp.length).collect();

        let num_runs = size_set.len() + if self.allow_mixed_stock_sizes { 1 } else { 0 };
        let callback = |progress| {
            progress_callback(progress / num_runs as f64);
        };

        let mut best_result = if self.allow_mixed_stock_sizes {
            // Optimize with all stock sizes
            self.optimize_with_stock_pieces::<BasicBin, _>(&self.stock_pieces.clone(), &callback)
        } else {
            // We're not allowing mixed sizes so just give an error result
            // here. Each stock size will be optimized separately below.
            // Note: it's safe to assume `self.cut_pieces` isn't empty because
            // that's checked at the beginning of this function.
            Err(no_fit_for_cut_piece_error(&self.cut_pieces[0]))
        };

        // Optimize each stock size separately and see if any have better result than
        // when optimizing with all stock sizes.
        for (i, length) in size_set.iter().enumerate() {
            let stock_pieces: Vec<StockPiece> = self
                .stock_pieces
                .iter()
                .filter(|sp| sp.length == *length)
                .cloned()
                .collect();

            let completed_runs = i + 1;
            if let Ok(solution) =
                self.optimize_with_stock_pieces::<BasicBin, _>(&stock_pieces, &|progress| {
                    progress_callback((completed_runs as f64 + progress) / num_runs as f64);
                })
            {
                match best_result {
                    Ok(ref best_solution) => {
                        // Use the lower-priced solution, but if the prices are the same, use the
                        // solution with the higher fitness score.
                        if solution.fitness < 0.0 || best_solution.fitness < 0.0 {
                            if solution.fitness > best_solution.fitness {
                                best_result = Ok(solution);
                            }
                        } else if solution.price < best_solution.price
                            || (solution.price == best_solution.price
                                && solution.fitness > best_solution.fitness)
                        {
                            best_result = Ok(solution);
                        }
                    }
                    Err(_) => best_result = Ok(solution),
                }
            }
        }

        if let Ok(ref mut solution) = &mut best_result {
            solution
                .stock_pieces
                .sort_by_key(|p| cmp::Reverse(p.length));
        };

        best_result
    }

    fn optimize_with_stock_pieces<B, F>(
        &self,
        stock_pieces: &[StockPiece],
        progress_callback: &F,
    ) -> Result<Solution>
    where
        B: Bin + Clone + Send + Into<ResultStockPiece>,
        F: Fn(f64),
    {
        let cut_pieces: Vec<&CutPieceWithId> = self.cut_pieces.iter().collect();

        let units: Vec<OptimizerUnit<B>> = OptimizerUnit::generate_initial_units(
            stock_pieces,
            cut_pieces,
            self.cut_width,
            self.random_seed,
        )?;

        let population_size = units.len();
        let mut result_units = Population::new(units)
            .set_size(population_size)
            .set_rand_seed(self.random_seed)
            .set_breed_factor(0.5)
            .set_survival_factor(0.6)
            .epochs(100, progress_callback)
            .finish();

        let best_unit = &mut result_units[0];
        if !best_unit.unused_cut_pieces.is_empty() {
            return Err(no_fit_for_cut_piece_error(
                best_unit.unused_cut_pieces.iter().next().unwrap(),
            ));
        }

        let fitness = best_unit.fitness();
        let price = best_unit.bins.iter().map(|bin| bin.price()).sum();

        let used_stock_pieces: Vec<ResultStockPiece> =
            best_unit.bins.drain(..).map(Into::into).collect();

        Ok(Solution {
            fitness,
            stock_pieces: used_stock_pieces,
            price,
        })
    }
}
