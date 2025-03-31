import os
import sys
import pickle
import pywt
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

# Add parent directory to sys.path if needed
parent_dir = os.path.join(os.getcwd(), '..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# These are presumably your local modules/utilities:
# Adjust imports to your file structure as needed
from ttknn.light_utility import Utility
from IPython.display import clear_output
import geobleu
from mkit.torch_support.tensor_utils import sequential_x_y_split, xy_to_tensordataset
from module.utility import LabelEncoder

# ---------------------------------
# 1. Wavelet Denoising Function
# ---------------------------------
def wavelet_denoise(data, wavelet='db4', level=None, alpha=1.0):
    """Denoise 1D data using wavelet thresholding."""
    coeffs = pywt.wavedec(data, wavelet, mode="per", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = alpha * sigma * np.sqrt(2 * np.log(len(data))) 
    # Soft-threshold detail coefficients
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")

# ---------------------------------
# 2. Model Definition
# ---------------------------------
class SingleNN(nn.Module):
    """
    Example MLP that uses embeddings for input tokens.
    window_size: sequence length
    embed_dim: dimensionality of embeddings
    vocab: size of the vocabulary
    """
    def __init__(self, window_size, embed_dim, vocab):
        super(SingleNN, self).__init__()
        self.embed = nn.Embedding(vocab, embed_dim)
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Linear(embed_dim * window_size, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.Tanh(),
            nn.Linear(32, vocab),
        )
        self.vocab = vocab - 1

    def forward(self, x):
        x = torch.clamp(x, min=0,  max=self.vocab)

        # x shape: (batch_size, window_size)
        x = self.embed(x)              # (batch_size, window_size, embed_dim)
        x = x.reshape(len(x), -1)      # (batch_size, window_size*embed_dim)
        x = self.net(x)                # (batch_size, vocab)
        return x

# ---------------------------------
# 3. Predict-Sequence Function
# ---------------------------------
def predict_sequence(model, initial_tokens, ahead, device):
    """
    Generate 'ahead' tokens from the 'model' given an initial token sequence.
    """
    in_x = torch.tensor(initial_tokens, dtype=torch.int64).to(device).unsqueeze(0)
    print("Initial in_x shape:", in_x.shape)
    
    sequences = []
    for _ in tqdm(range(ahead)):
        out = model(in_x)  # (batch_size=1, vocab)
        out_token = torch.argmax(out, dim=1).unsqueeze(0)  # (1, 1)
        # Shift by 1 position: concat new token, drop oldest
        in_x = torch.concat([in_x, out_token], dim=1)[:, 1:]
        sequences.append(out_token[0].item())
    return sequences

# ---------------------------------
# 4. Training Helper
# ---------------------------------
def get_model(df, look_back, col='x', vocab_size=40401, epochs=20, device=torch.device("cuda")):
    """
    Trains a SingleNN model on the specified column of tmp_train_df (col).
    Uses sequential_x_y_split and returns the model plus the final window segment.
    """
    train_x, train_y = sequential_x_y_split(
        df[col].values,
        look_back=look_back
    )
    train_y = train_y.ravel()

    model = SingleNN(window_size=train_x.shape[1], embed_dim=100, vocab=vocab_size).to(device)
    train_loader = xy_to_tensordataset(train_x, train_y, return_loader=True, shuffle=True)

    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # For tracking loss if needed
    losses = []

    for epoch in range(epochs):
        avg_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, dtype=torch.int64)
            y_batch = y_batch.to(device, dtype=torch.int64)

            optimizer.zero_grad()
            outputs = model(x_batch)
            outputs = torch.clamp(outputs, min=0, max=vocab_size - 1)
            y_batch = torch.clamp(y_batch, min=0, max=vocab_size - 1)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        avg_loss /= len(train_loader)
        losses.append(avg_loss)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    final_segment = train_x[-1]  # The last input window
    return model, final_segment

# ---------------------------------
# 5. Main Pipeline
# ---------------------------------
if __name__ == "__main__":
    # Load Data
    df = pd.read_csv("../cityD-dataset.csv")

    # Wavelet Denoise
    ax_series = df['x'].values
    ay_series = df['y'].values
    ax_denoised = wavelet_denoise(ax_series, wavelet='db4', level=5, alpha=5)[:len(ax_series)]
    ay_denoised = wavelet_denoise(ay_series, wavelet='db4', level=5, alpha=5)[:len(ay_series)]
    
    df['denoised_x'] = np.round(ax_denoised).astype(int)
    df['denoised_y'] = np.round(ay_denoised).astype(int)

    # Constants
    SPLIT_DATE = 60
    END_DATE = 75
    NUM_OF_TIMESTAMPS = 48
    VOCAB = 201
    LENGTHS = [NUM_OF_TIMESTAMPS * 7]  # Just an example
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dictionary to store results
    results = {}

    # Go through each user
    for uid in df.uid.unique():
        uid_df = df[df.uid == uid]
        print(uid)

        # Split into train/test by date
        train_df = uid_df[uid_df.d < SPLIT_DATE].copy()
        test_df  = uid_df[uid_df.d >= SPLIT_DATE].copy()

        # Prepare a label encoder, if needed
        encoder = LabelEncoder()

        # Prepare train template (full grid of [uid, day, t]) to ensure continuity
        train_template_idx = pd.MultiIndex.from_product(
            [train_df.uid.unique(), range(SPLIT_DATE), range(NUM_OF_TIMESTAMPS)]
        )
        train_template = pd.DataFrame(index=train_template_idx).reset_index()
        train_template.columns = ['uid', 'd', 't']

        # Merge to fill missing rows
        train_df = train_template.merge(train_df, on=['uid','d','t'], how='left')
        train_df = train_df.fillna(method='ffill').fillna(method='bfill')

        # Test template (future horizon)
        test_template_idx = pd.MultiIndex.from_product(
            [range(SPLIT_DATE, END_DATE), range(NUM_OF_TIMESTAMPS)]
        )
        test_template = pd.DataFrame(index=test_template_idx).reset_index()
        test_template.columns = ['d','t']

        # Number of tokens to predict in the future
        AHEAD = test_template.d.nunique() * test_template.t.nunique()
        print(AHEAD)
        # We'll run two experiments: Original vs. Denoised
        experiment_scores = {}

        for data_version in ["original", "denoised"]:
            if data_version == "original":
                tmp_train_df = train_df[['x','y']].rename(columns={'x':'x','y':'y'})
                tmp_test_cols = ['x','y']
            else:
                # Using denoised columns
                tmp_train_df = train_df[['denoised_x','denoised_y']].rename(
                    columns={'denoised_x': 'x','denoised_y': 'y'}
                )
                tmp_test_cols = ['denoised_x','denoised_y']

            # Apply encoding (if needed) – transforms each row of (x,y) to integer tokens
   
            # Train X-model
            x_model, x_final_segment = get_model(
                tmp_train_df,
                look_back=LENGTHS[0],
                col='x',
                vocab_size=VOCAB,
                epochs=EPOCHS,
                device=DEVICE
            )

            # Train Y-model
            y_model, y_final_segment = get_model(
                tmp_train_df,
                look_back=LENGTHS[0],
                col='y',
                vocab_size=VOCAB,
                epochs=EPOCHS,
                device=DEVICE
            )

            # Predict X-sequence
            x_segments = predict_sequence(
                model=x_model,
                initial_tokens=x_final_segment,
                ahead=AHEAD,
                device=DEVICE
            )

            # Predict Y-sequence
            y_segments = predict_sequence(
                model=y_model,
                initial_tokens=y_final_segment,
                ahead=AHEAD,
                device=DEVICE
            )
            clear_output(wait=True)
            # Merge predictions with test template
            tmp_template = test_template.copy()
            tmp_template['x'] = x_segments
            tmp_template['y'] = y_segments

            # Re-align to the actual test rows
            tmp_template = test_df[['uid','d','t']].merge(tmp_template, on=['d','t'], how='left')

            # Evaluate using DTW or your metric of choice
            # Utility.to_eval_format might reformat df to lat/lon/time or similar
            dtw_score = geobleu.calc_dtw(
                Utility.to_eval_format(tmp_template),
                Utility.to_eval_format(test_df)
            )

            experiment_scores[data_version] = dtw_score

        # Store results in dictionary
        results[uid] = {
            'dtw_original': experiment_scores['original'],
            'dtw_denoised': experiment_scores['denoised']
        }

        # ---------------------------------
        # 6. Pickle the Results
        # ---------------------------------
        output_path = "results.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"Finished! Results saved to {output_path}")
        # print("Results dictionary:", results)
