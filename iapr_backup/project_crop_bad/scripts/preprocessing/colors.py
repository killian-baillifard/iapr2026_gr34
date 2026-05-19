import cv2
import numpy as np
from matplotlib import pyplot as plt
from project_crop_bad.scripts.dataset import load_manually_segmented_images

def circular_mean_std(angles, max_val=180):
    """Compute mean and stddev for circular data (e.g. hue)."""
    radians = angles * (2 * np.pi / max_val)
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    mean = np.arctan2(sin_mean, cos_mean) * (max_val / (2 * np.pi))
    if mean < 0:
        mean += max_val
    # Circular stddev
    R = np.sqrt(sin_mean**2 + cos_mean**2)  # mean resultant length
    std = np.sqrt(-2 * np.log(R)) * (max_val / (2 * np.pi))
    return mean, std

if __name__ == "__main__":

    # Load manually segmented images
    images = load_manually_segmented_images()

    # Transform into HSV color space with an alpha mask
    hsv = {}
    alpha_mask = {}
    for color, image in images.items():
        alpha_mask[color] = image[:, :, 3] != 0
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        hsv[color] = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Compute color distributions in HSV space, omitting pixels outside the alpha mask
    h_distribution = {}
    s_distribution = {}
    v_distribution = {}
    for color in images.keys():
        h_distribution[color] = hsv[color][:, :, 0][alpha_mask[color]]
        s_distribution[color] = hsv[color][:, :, 1][alpha_mask[color]]
        v_distribution[color] = hsv[color][:, :, 2][alpha_mask[color]]

    # Compute mean and standard deviation in HSV space
    h_mean = {}
    s_mean = {}
    v_mean = {}
    h_stddev = {}
    s_stddev = {}
    v_stddev = {}
    for color in images.keys():
        h_mean[color], h_stddev[color] = circular_mean_std(h_distribution[color])
        s_mean[color] = np.mean(s_distribution[color])
        v_mean[color] = np.mean(v_distribution[color])
        s_stddev[color] = np.std(s_distribution[color])
        v_stddev[color] = np.std(v_distribution[color])

    # Color display names and matplotlib colors for each label
    color_display = {
        "r": ("Red",    "red"),
        "y": ("Yellow", "gold"),
        "g": ("Green",  "green"),
        "b": ("Blue",   "blue"),
        "k": ("Black",  "black"),
    }

    # HSV channel configs: (name, distributions dict, means dict, stddevs dict, x range)
    channels = [
        ("Hue",        h_distribution, h_mean, h_stddev, (0,   180)),
        ("Saturation", s_distribution, s_mean, s_stddev, (0,   256)),
        ("Value",      v_distribution, v_mean, v_stddev, (0,   256)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.canvas.manager.set_window_title("Colors distributions")

    for ax, (channel_name, dist_dict, mean_dict, std_dict, (xmin, xmax)) in zip(axes, channels):
        for color, (label, mpl_color) in color_display.items():
            data = dist_dict[color]
            mu   = mean_dict[color]
            sig  = std_dict[color]

            label = f"{label} - N({int(np.round(mu))}, {int(np.round(sig))})"

            # Normalized histogram
            counts, bin_edges = np.histogram(data, bins=xmax - xmin, range=(xmin, xmax))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            counts = counts / counts.sum()  # normalize to probability
            ax.plot(bin_centers, counts, color=mpl_color, alpha=0.5, label=label)

            # Gaussian fit (dashed)
            if sig > 0:
                x = np.linspace(xmin, xmax, 500)
                # For hue: sum wrapped Gaussians to handle the 0/180 boundary
                if channel_name == "Hue":
                    gaussian = (
                        np.exp(-0.5 * ((x - mu) / sig) ** 2) +
                        np.exp(-0.5 * ((x - mu + xmax) / sig) ** 2) +
                        np.exp(-0.5 * ((x - mu - xmax) / sig) ** 2)
                    )
                else:
                    gaussian = np.exp(-0.5 * ((x - mu) / sig) ** 2)
                gaussian = gaussian / gaussian.max() * counts.max()
                ax.plot(x, gaussian, color=mpl_color, linestyle="--", linewidth=1.5)

        ax.set_title(f"{channel_name} distribution")
        ax.set_xlabel(channel_name)
        ax.set_ylabel("Normalized frequency")
        ax.set_xlim(xmin, xmax)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
