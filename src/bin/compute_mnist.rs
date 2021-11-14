use deep_learning_playground::neural_network::{self, activate_functions};
use deep_learning_playground::setup::dlfs::chap3;
use deep_learning_playground::setup::mnist::{batched, load_data, test_dataset, Batched};
use deep_learning_playground::utils::natural_transform::to_io;
use std::env;
use std::io;
use std::time::{Duration, Instant};
use std::vec::Vec;

fn compute(td: Batched, trained_d