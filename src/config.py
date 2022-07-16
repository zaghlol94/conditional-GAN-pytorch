config = {
    "LEARNING_RATE": 1e-4,  # could also use two lrs, one for gen and one for disc
    "BATCH_SIZE": 64,
    "IMAGE_SIZE": 64,
    "CHANNELS_IMG": 1,
    "NOISE_DIM": 100,
    "NUM_EPOCHS": 5,
    "FEATURES_DISC": 64,
    "FEATURES_GEN": 64,
    "CRITIC_ITERATIONS": 5,
    "WEIGHT_CLIP": 0.01,
    "LAMBDA_GP": 10,
    "NUM_CLASSES": 10,
    "GEN_EMBEDDING": 100,
}
