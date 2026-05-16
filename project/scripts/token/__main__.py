from project.scripts.dataset import load_train_images
from project.scripts.token import detect_active_player

if __name__ == "__main__":

    images, labels = load_train_images()

    print("\n=== Detection ===")
    predictions = detect_active_player(images, debug=False)

    print("\n=== Results ===")
    correct  = sum(str(p) == str(l.active_player)
                   for p, l in zip(predictions, labels) if p is not None)
    detected = sum(p is not None for p in predictions)
    print(f"  Detected : {detected}/{len(images)}")
    print(f"  Correct  : {correct}/{len(images)}")
    for i, (p, l) in enumerate(zip(predictions, labels)):
        status = "✓" if str(p) == str(l.active_player) else "✗"
        print(f"  {status} Image {i}: predicted={p}  truth={l.active_player}")
