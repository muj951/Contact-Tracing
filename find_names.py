from src.preprocessing import load_data
from src.train import get_latest_model

def find_people_with_contacts():
    # 1. Load your data and the trained model
    df = load_data()
    model = get_latest_model()

    # 2. Assign the labels from the model to your data
    df['cluster'] = model.labels_

    # 3. Filter out the noise (-1) and find the people in actual clusters
    people_with_contacts = df[df['cluster'] != -1]['id'].unique()

    if len(people_with_contacts) > 0:
        print("\n✅ Found people with contacts! Try testing one of these names in FastAPI:")
        # Print the first 10 names found
        for name in people_with_contacts[:10]:
            print(f" - {name}")
    else:
        print("\n⚠️ Wow! With an epsilon of 0.0018288, absolutely everyone is isolated (no contacts).")
        print("You might need to increase epsilon in train.py and re-train to get clusters.")

if __name__ == "__main__":
    find_people_with_contacts()
