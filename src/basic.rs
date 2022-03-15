use super::*;

use std::borrow::Borrow;

#[derive(Clone, Debug)]
pub(crate) struct BasicBin {
    length: usize,
    blade_width: usize,
    cut_pieces: Vec<UsedCutPiece>,
    price: usize,
}

impl Bin for BasicBin {
    fn new(length: usize, blade_width: usize, price: usize) -> Self {
        BasicBin {
            length,
            blade_width,
            cut_pieces: Default::default(),
            price,
        }
    }

    fn fitness(&self) -> f64 {
        let used_length = self
            .cut_pieces
            .iter()
            .fold(0, |acc, p| acc + (p.end - p.start) as u64) as f64;

        (used_length / self.length as f64).powf(2.0)
    }

    fn price(&self) -> usize {
        self.price
    }

    fn remove_cut_pieces<I>(&mut self, cut_pieces: I) -> usize
    where
        I: Iterator,
        I::Item: Borrow<UsedCutPiece>,
    {
        let old_len = self.cut_pieces.len();
        for cut_piece_to_remove in cut_pieces {
            for i in (0..self.cut_pieces.len()).rev() {
                if &self.cut_pieces[i] == cut_piece_to_remove.borrow() {
                    self.cut_pieces.remove(i);
                }
            }
        }
        self.compact();
        old_len - self.cut_pieces.len()
    }

    fn cut_pieces(&self) -> std::slice::Iter<'_, UsedCutPiece> {
        self.cut_pieces.iter()
    }

    fn insert_cut_piece(&mut self, cut_piece: &CutPieceWithId) -> bool {
        self.insert_cut_piece(cut_piece)
    }

    fn matches_stock_piece(&self, stock_piece: &StockPiece) -> bool {
        self.length == stock_piece.length && self.price == stock_piece.price
    }
}

impl BasicBin {
    /// Insert cut piece in bin if it fits. Returns `true` if inserted.
    fn insert_cut_piece(&mut self, cut_piece: &CutPieceWithId) -> bool {
        if let Some(insertion_point) = self.insertion_point(cut_piece.length) {
            let start = insertion_point;
            let end = start + cut_piece.length;
            let used_piece = UsedCutPiece {
                id: cut_piece.id,
                external_id: cut_piece.external_id,
                start,
                end,
            };

            self.cut_pieces.push(used_piece);

            true
        } else {
            false
        }
    }

    /// Where the next piece should be inserted if it fits.
    fn insertion_point(&self, cut_length: usize) -> Option<usize> {
        let insertion_point = if self.cut_pieces.is_empty() {
            0
        } else {
            self.cut_pieces[self.cut_pieces.len() - 1].end + self.blade_width
        };

        if insertion_point < self.length && cut_length <= self.length - insertion_point {
            Some(insertion_point)
        } else {
            None
        }
    }

    /// Push cut pieces to one side to fill in any gaps that are larger than the blade width.
    fn compact(&mut self) {
        let mut prev_end = 0;
        for cut_piece in self.cut_pieces.iter_mut() {
            let blade_width_adjustment = if prev_end > 0 { self.blade_width } else { 0 };
            if cut_piece.start - prev_end > blade_width_adjustment {
                let length = cut_piece.length();
                cut_piece.start = prev_end + blade_width_adjustment;
                cut_piece.end = cut_piece.start + length;
            }
            prev_end = cut_piece.end;
        }
    }
}

impl From<BasicBin> for ResultStockPiece {
    fn from(bin: BasicBin) -> Self {
        Self {
            length: bin.length,
            cut_pieces: bin.cut_pieces.into_iter().map(Into::into).collect(),
            price: bin.price,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_cut_pieces() {
        let cut_pieces = &[
            CutPieceWithId {
                id: 0,
                external_id: None,
                length: 10,
            },
            CutPieceWithId {
                id: 1,
                external_id: None,
                length: 10,
            },
            CutPieceWithId {
                id: 2,
                external_id: None,
                length: 10,
            },
            CutPieceWithId {
                id: 3,
                external_id: None,
                length: 10,
            },
        ];

        let mut bin = BasicBin::new(96, 1, 0);
        cut_pieces.iter().for_each(|cut_piece| {
            bin.insert_cut_piece(cut_piece);
        });

        assert_eq!(bin.cut_pieces().len(), 4);

        let cut_pieces_to_remove = [
            UsedCutPiece {
                id: 1,
                external_id: None,
                start: 0,
                end: 0,
            },
            UsedCutPiece {
                id: 3,
                external_id: None,
                start: 0,
                end: 0,
            },
        ];

        bin.remove_cut_pieces(cut_pieces_to_remove.iter());

        assert_eq!(bin.cut_pieces().len(), 2);
        assert_eq!(bin.cut_pieces().next().unwrap().id, 0);
        assert_eq!(bin.cut_pieces().nth(1).unwrap().id, 2);
    }

    #[test]
    fn bin_matches_stock_piece() {
        let bin = BasicBin {
            length: 96,
            blade_width: 1,
            cut_pieces: Default::default(),
            price: 0,
        };

        let stock_piece = StockPiece {
            length: 96,
            price: 0,
            quantity: Some(20),
        };

        assert!(bin.matches_stock_piece(&stock_piece));
    }

    #[test]
    fn bin_does_not_match_stock_pieces() {
        let bin = BasicBin {
            length: 96,
            blade_width: 1,
            cut_pieces: Default::default(),
            price: 0,
        };

        let stock_pieces = &[
            StockPiece {
                length: 96,
                price: 1,
                quantity: Some(1),
            },
            StockPiece {
                length: 10,
                price: 0,
                quantity: Some(2),
            },
            StockPiece {
                length: 97,
                price: 0,
                quantity: Some(3),
            },
            StockPiece {
                length: 96,
                price: 10,
                quantity: None,
            },
        ];

        stock_pieces
            .iter()
            .for_each(|stock_piece| assert!(!bin.matches_stock_piece(&stock_piece)))
    }
}
