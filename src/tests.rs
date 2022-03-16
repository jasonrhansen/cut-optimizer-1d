use super::*;

static STOCK_PIECES: &[StockPiece] = &[
    StockPiece {
        length: 96,
        price: 0,
        quantity: None,
    },
    StockPiece {
        length: 120,
        price: 0,
        quantity: None,
    },
];

static CUT_PIECES: &[CutPiece] = &[
    CutPiece {
        quantity: 1,
        external_id: Some(1),
        length: 30,
    },
    CutPiece {
        quantity: 1,
        external_id: Some(2),
        length: 30,
    },
    CutPiece {
        quantity: 1,
        external_id: Some(3),
        length: 30,
    },
    CutPiece {
        quantity: 1,
        external_id: Some(4),
        length: 30,
    },
];

fn sanity_check_solution(solution: &Solution, num_cut_pieces: usize) {
    let stock_pieces = &solution.stock_pieces;

    assert!(solution.fitness <= 1.0);

    // The number of result cut pieces should match the number of input cut pieces.
    assert_eq!(
        stock_pieces
            .iter()
            .map(|sp| sp.cut_pieces.len())
            .sum::<usize>(),
        num_cut_pieces
    );

    for stock_piece in stock_pieces {
        let cut_piece_total_length = stock_piece
            .cut_pieces
            .iter()
            .map(|cp| cp.end - cp.start)
            .sum::<usize>();

        // Make sure the stock piece is big enough for the cut pieces and waste pieces.
        assert!(stock_piece.length >= cut_piece_total_length);

        // Assert that start isn't larger than end for all cut pieces
        for stock_piece in &stock_piece.cut_pieces {
            assert!(stock_piece.start <= stock_piece.end);
        }

        // Assert that cut_pieces don't overlap
        for pair in stock_piece.cut_pieces.windows(2) {
            assert!(pair[0].end <= pair[1].start);
        }
    }
}

#[test]
fn optimize() {
    let solution = Optimizer::new()
        .add_stock_pieces(STOCK_PIECES.iter().cloned().collect::<Vec<_>>())
        .add_cut_pieces(CUT_PIECES.iter().cloned().collect::<Vec<_>>())
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, CUT_PIECES.len());
}

#[test]
fn optimize_non_fitting_cut_piece() {
    let result = Optimizer::new()
        .add_stock_piece(StockPiece {
            length: 10,
            quantity: None,
            price: 0,
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: Some(1),
            length: 11,
        })
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {});

    assert!(
        matches!(result, Err(Error::NoFitForCutPiece(_))),
        "should have returned Error::NoFitForCutPiece"
    )
}

#[test]
fn optimize_no_allow_mixed_stock_sizes() {
    let solution = Optimizer::new()
        .add_stock_pieces(STOCK_PIECES.iter().cloned().collect::<Vec<_>>())
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: Some(1),
            length: 96,
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: Some(2),
            length: 120,
        })
        .set_cut_width(1)
        .set_random_seed(1)
        .allow_mixed_stock_sizes(false)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, 2);

    assert_eq!(solution.stock_pieces.len(), 2);
    for stock_piece in solution.stock_pieces {
        // Since we aren't allowing mixed sizes,
        // all stock pieces will need to be 120 long.
        assert_eq!(stock_piece.length, 120)
    }
}

#[test]
fn optimize_different_stock_piece_prices() {
    let solution = Optimizer::new()
        .add_stock_piece(StockPiece {
            length: 96,
            price: 1,
            quantity: None,
        })
        .add_stock_piece(StockPiece {
            length: 120,
            // Maker the 48x120 stock piece more expensive than (2) 48x96 pieces.
            price: 3,
            quantity: None,
        })
        .add_cut_piece(CutPiece {
            quantity: 2,
            external_id: Some(1),
            length: 50,
        })
        .set_cut_width(1)
        .set_random_seed(1)
        .allow_mixed_stock_sizes(false)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, 2);

    // A single 120 stock piece could be used, but since we've set (2) 96 pieces to
    // be a lower price than (1) 120, it should use (2) 96 pieces instead.
    assert_eq!(solution.stock_pieces.len(), 2);
    for stock_piece in solution.stock_pieces {
        assert_eq!(stock_piece.length, 96)
    }
}

#[test]
fn optimize_same_stock_piece_prices() {
    let solution = Optimizer::new()
        .add_stock_piece(StockPiece {
            length: 96,
            price: 0,
            quantity: None,
        })
        .add_stock_piece(StockPiece {
            length: 120,
            price: 0,
            quantity: None,
        })
        .add_cut_piece(CutPiece {
            quantity: 2,
            external_id: Some(1),
            length: 50,
        })
        .set_cut_width(1)
        .set_random_seed(1)
        .allow_mixed_stock_sizes(false)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, 2);

    assert_eq!(solution.stock_pieces.len(), 1);
    assert_eq!(solution.stock_pieces[0].length, 120)
}

#[test]
fn optimize_stock_quantity_too_low() {
    let result = Optimizer::new()
        .add_stock_piece(StockPiece {
            length: 96,
            price: 0,
            quantity: Some(1),
        })
        .add_cut_piece(CutPiece {
            quantity: 2,
            external_id: None,
            length: 96,
        })
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {});

    assert!(
        result.is_err(),
        "should fail because stock quantity is too low"
    );
}

#[test]
fn optimize_stock_quantity() {
    let solution = Optimizer::new()
        .add_stock_piece(StockPiece {
            length: 96,
            price: 0,
            quantity: Some(2),
        })
        .add_cut_piece(CutPiece {
            quantity: 2,
            external_id: None,
            length: 96,
        })
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, 2);
}

#[test]
fn optimize_stock_quantity_multiple() {
    let solution = Optimizer::new()
        .add_stock_piece(StockPiece {
            length: 96,
            price: 0,
            quantity: Some(2),
        })
        .add_stock_piece(StockPiece {
            length: 192,
            price: 0,
            quantity: Some(1),
        })
        .add_cut_piece(CutPiece {
            quantity: 2,
            external_id: None,
            length: 96,
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: None,
            length: 192,
        })
        .set_cut_width(0)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, 3);
}

#[test]
fn optimize_one_stock_piece_several_cut_pieces() {
    let solution = Optimizer::new()
        .add_stock_piece(StockPiece {
            length: 96,
            price: 0,
            quantity: Some(1),
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: None,
            length: 8,
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: None,
            length: 9,
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: None,
            length: 10,
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: None,
            length: 12,
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: None,
            length: 13,
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: None,
            length: 14,
        })
        .add_cut_piece(CutPiece {
            quantity: 1,
            external_id: None,
            length: 15,
        })
        .set_cut_width(0)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, 7);
}

#[test]
fn optimize_stock_duplicate_cut_piece() {
    let solution = Optimizer::new()
        .add_stock_piece(StockPiece {
            length: 96,
            price: 0,
            quantity: Some(1),
        })
        .add_stock_piece(StockPiece {
            length: 192,
            price: 0,
            quantity: Some(1),
        })
        .add_cut_piece(CutPiece {
            quantity: 2,
            external_id: None,
            length: 96,
        })
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, 2);
}

#[test]
fn optimize_32_cut_pieces_on_1_stock_piece() {
    let mut optimizer = Optimizer::new();
    optimizer.add_stock_piece(StockPiece {
        length: 351,
        price: 0,
        quantity: None,
    });

    let num_cut_pieces = 32;

    optimizer.add_cut_piece(CutPiece {
        quantity: num_cut_pieces,
        external_id: Some(1),
        length: 10,
    });

    let solution = optimizer
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, num_cut_pieces);

    let stock_pieces = solution.stock_pieces;
    assert_eq!(stock_pieces.len(), 1);
    let cut_pieces = &stock_pieces[0].cut_pieces;
    assert_eq!(cut_pieces.len(), 32);
}

#[test]
fn optimize_32_cut_pieces_on_2_stock_pieces_zero_cut_width() {
    let mut optimizer = Optimizer::new();
    optimizer.add_stock_piece(StockPiece {
        length: 160,
        price: 0,
        quantity: None,
    });

    let num_cut_pieces = 32;

    optimizer.add_cut_piece(CutPiece {
        quantity: num_cut_pieces,
        external_id: Some(1),
        length: 10,
    });

    let solution = optimizer
        .set_cut_width(0)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, num_cut_pieces);

    let stock_pieces = solution.stock_pieces;
    assert_eq!(stock_pieces.len(), 2);
    assert_eq!(stock_pieces[0].cut_pieces.len(), 16);
    assert_eq!(stock_pieces[1].cut_pieces.len(), 16);
}

#[test]
fn optimize_32_cut_pieces_on_2_stock_piece() {
    let mut optimizer = Optimizer::new();
    optimizer.add_stock_piece(StockPiece {
        length: 175,
        price: 0,
        quantity: None,
    });

    let num_cut_pieces = 32;

    optimizer.add_cut_piece(CutPiece {
        quantity: num_cut_pieces,
        external_id: Some(1),
        length: 10,
    });

    let solution = optimizer
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, num_cut_pieces);

    let stock_pieces = solution.stock_pieces;
    assert_eq!(stock_pieces.len(), 2);
}

#[test]
fn optimize_64_cut_pieces_on_2_stock_pieces() {
    let mut optimizer = Optimizer::new();
    optimizer.add_stock_piece(StockPiece {
        length: 352,
        price: 0,
        quantity: None,
    });

    let num_cut_pieces = 64;

    optimizer.add_cut_piece(CutPiece {
        quantity: num_cut_pieces,
        external_id: Some(1),
        length: 10,
    });

    let solution = optimizer
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, num_cut_pieces);

    let stock_pieces = solution.stock_pieces;
    assert_eq!(stock_pieces.len(), 2);
    assert_eq!(stock_pieces[0].cut_pieces.len(), 32);
    assert_eq!(stock_pieces[1].cut_pieces.len(), 32);
}

#[test]
fn optimize_random_cut_pieces() {
    let mut optimizer = Optimizer::new();
    optimizer.add_stock_piece(StockPiece {
        length: 96,
        price: 0,
        quantity: None,
    });
    optimizer.add_stock_piece(StockPiece {
        length: 120,
        price: 0,
        quantity: None,
    });

    let mut rng: StdRng = SeedableRng::seed_from_u64(1);

    let num_cut_pieces = 30;

    optimizer.add_cut_piece(CutPiece {
        quantity: num_cut_pieces,
        external_id: Some(1),
        length: rng.gen_range(1..=120),
    });

    let solution = optimizer
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    sanity_check_solution(&solution, num_cut_pieces);
}

#[test]
fn optimize_stock_quantity_1() {
    let mut optimizer = Optimizer::new();
    optimizer.add_stock_piece(StockPiece {
        quantity: Some(1),
        length: 96,
        price: 0,
    });

    optimizer.add_cut_piece(CutPiece {
        quantity: 1,
        external_id: Some(1),
        length: 50,
    });

    optimizer.add_cut_piece(CutPiece {
        quantity: 1,
        external_id: Some(1),
        length: 20,
    });

    let solution = optimizer
        .set_cut_width(1)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    assert_eq!(solution.stock_pieces.len(), 1);
    sanity_check_solution(&solution, 2);
}

#[test]
fn optimize_stock_quantity_2() {
    let mut optimizer = Optimizer::new();
    optimizer.add_stock_piece(StockPiece {
        quantity: Some(2),
        length: 200,
        price: 130,
    });

    optimizer.add_cut_piece(CutPiece {
        quantity: 6,
        external_id: Some(1),
        length: 50,
    });

    let solution = optimizer
        .set_cut_width(2)
        .set_random_seed(1)
        .optimize(|_| {})
        .unwrap();

    assert_eq!(solution.stock_pieces.len(), 2);
    sanity_check_solution(&solution, 6);
}

#[test]
fn deterministic_solutions() {
    // Run the same optimization multiple times with the same random seed and
    // check if the solution is the same each time.
    let solutions: Vec<Solution> = (0..10)
        .map(|_| {
            let plywood = StockPiece {
                quantity: Some(2),
                length: 200,
                price: 130,
            };

            let cut_piece_a = CutPiece {
                quantity: 6,
                external_id: Some(1),
                length: 50,
            };

            let mut optimizer = Optimizer::new();
            optimizer.add_stock_piece(plywood);
            optimizer.add_cut_piece(cut_piece_a);
            optimizer.set_cut_width(2);
            optimizer.set_random_seed(1);

            optimizer.optimize(|_| {}).unwrap()
        })
        .collect();

    solutions.windows(2).for_each(|window| {
        let solution1 = &window[0];
        let solution2 = &window[1];
        assert_eq!(solution1.fitness, solution2.fitness);
        assert_eq!(solution1.price, solution2.price);
        solution1
            .stock_pieces
            .iter()
            .zip(solution2.stock_pieces.iter())
            .for_each(|(stock_piece1, stock_piece2)| {
                assert_eq!(stock_piece1.length, stock_piece2.length);
                assert_eq!(stock_piece1.price, stock_piece2.price);
                stock_piece1
                    .cut_pieces
                    .iter()
                    .zip(stock_piece2.cut_pieces.iter())
                    .for_each(|(cut_piece1, cut_piece2)| {
                        assert_eq!(cut_piece1, cut_piece2);
                    });
            })
    });
}
