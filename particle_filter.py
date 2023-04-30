import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(0)


# Initialize Parameters:
SIGMA_LOCATION = 1
SIGMA_VELOCITY = 0.8
MU = 0
# change IDs to your IDs.
ID1 = "308345891"
ID2 = "211670849"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y


def predict_particles(s_prior: np.ndarray, initial_Flag=False) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = s_prior
    if not initial_Flag:
        state_drifted[0:2, :] = s_prior[0:2, :] + s_prior[4:6, :]
    # Add Noise
    state_drifted[0:2, :] += np.round(np.random.normal(MU, SIGMA_LOCATION, size=(2, N)))
    state_drifted[4:6, :] += np.round(np.random.normal(MU, SIGMA_VELOCITY, size=(2, N)))
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((1, 16 * 16 * 16))
    """ DELETE THE LINE ABOVE AND:INSERT YOUR CODE HERE."""
    X_C, Y_C = state[0], state[1]
    helf_width, helf_height = state[2], state[3]
    # Take intrest part from the image and quantize
    I_subportion = image[Y_C - helf_height:Y_C + helf_height + 1, X_C - helf_width: X_C + helf_width + 1]
    I_subportion = np.round(I_subportion / 16).astype(np.uint8)
    hist, _ = np.histogramdd(
        I_subportion.reshape(-1, 3),
        bins=(16, 16, 16),
        range=((0, 16), (0, 16), (0, 16))
    )
    hist = hist.reshape((4096, 1))
    hist /= np.sum(hist)
    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    r = np.random.uniform(low=0.0, high=1.0, size=(N,))
    J = np.searchsorted(cdf, r, side='left')
    S_next = previous_state[:, J]
    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    distance = np.exp(20 * np.sum(np.sqrt(p * q)))
    return distance

def calc_W_C(image, S, q):
    """This function calculate the weights and the CDF vectors
      Args:
        S: np.ndarray matrix. Matrix of particle States.
        q: np.ndarray. second histogram.

    Return:
        W: np.ndarray matrix. The particle weights.
        C: np.ndarray matrix. The particle CDF.
    """
    W = np.zeros((N, ))
    for i in range(S.shape[1]):
        state = S[:, i]
        p = compute_normalized_histogram(image, state)
        W[i] = bhattacharyya_distance(p, q)
    W /= np.sum(W)
    C = np.cumsum(W)
    return W, C



def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:, :, ::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Avg particle box
    (x_avg, y_avg, w_avg, h_avg) = (int(np.sum(state[0, :] * W)), int(np.sum(state[1, :] * W)), int(np.sum(state[2, :] * W)), int(np.sum(state[3, :] * W)))

    rect = patches.Rectangle((x_avg-w_avg, y_avg-h_avg), 2.2*w_avg, 2.2*h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    index = np.argmax(W)
    (x_max, y_max, w_max, h_max) = (int(state[0, index]), int(state[1, index]), int(state[2, index]), int(state[3, index]))

    rect = patches.Rectangle((x_max-w_max, y_max-h_max), 2.2*w_max, 2.2*h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame, initial_Flag=True)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    """INSERT YOUR CODE HERE."""
    W, C = calc_W_C(image, S, q)
    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:
        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        """INSERT YOUR CODE HERE."""
        W, C = calc_W_C(current_image, S, q)
        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed % 10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
