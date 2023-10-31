import tensorflow as tf
from transformers import BartTokenizer, TFBartForConditionalGeneration
from sklearn.model_selection import train_test_split

# Initialize the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Sample dataset (replace with your own data)
data = {
    "text": [
        "The Israel-Palestine conflict is a protracted and deeply rooted conflict between Israelis and Palestinians. Here, we present a brief overview of this complex issue. The roots of the Israel-Palestine conflict can be traced back to the late 19th and early 20th centuries when nationalist movements among both Jewish and Arab communities emerged. In 1948, the United Nations proposed a plan to partition the land into separate Jewish and Arab states, leading to the establishment of Israel. The Arab world rejected this partition, sparking the first Arab-Israeli war.The primary issues at the heart of the conflict include territorial disputes over areas like the West Bank and Gaza Strip, the right of return for Palestinian refugees, security concerns, and the status of Jerusalem. These disputes have resulted in numerous conflicts and diplomatic standstills. Periodic conflicts in the Gaza Strip have led to significant loss of life and damage, with intermittent ceasefire agreements. While peace talks have been attempted, a lasting resolution remains elusive. Recent developments include normalization agreements between Israel and several Arab nations, reshaping regional dynamics.The Israel-Palestine conflict has had a severe humanitarian impact, leading to the suffering of both Israelis and Palestinians. The well-being of civilians in conflict areas has been a major concern for international organizations.The Israel-Palestine conflict is a deeply rooted issue with a long history. Achieving a lasting solution will require international cooperation, diplomacy, and dialogue. The road to peace is challenging, but it remains a vital pursuit for the future of the region."
        " A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy. The region of phase space is specified in which the calculation is most reliable. Good agreement is demonstrated with data from the Fermilab Tevatron, and predictions are made for more detailed tests with CDF and DO data. Predictions are shown for distributions of diphoton pairs produced at the energy of the Large Hadron Collider (LHC). Distributions of the diphoton pairs from the decay of a Higgs boson are contrasted with those produced from QCD processes at the LHC, showing that enhanced sensitivity to the signal can be obtained with judicious selection of events."
        "The benefits of exercise for a healthy lifestyle",
        "New technology trends in 2023",
        "The importance of a balanced diet",
        "Travel tips for a memorable vacation",
        "How to stay productive while working from home",
    ],
    "title": [
        " Israel-Palestine Conflict: A Longstanding Struggle for Peace"
        " Calculation of prompt diphoton production cross sections at Tevatron and LHC energies"
        "Exercise for a Healthy Lifestyle",
        "2023 Technology Trends",
        "The Importance of a Balanced Diet",
        "Tips for a Memorable Vacation",
        "Productivity Tips for Remote Work",
    ],
}

# Separate the dataset into input texts and target titles
input_texts = data["text"]
target_titles = data["title"]

# Tokenize input texts and target titles
input_encodings = tokenizer(input_texts, truncation=True, padding=True, return_tensors="tf")
target_encodings = tokenizer(target_titles, truncation=True, padding=True, return_tensors="tf")

# Split the dataset into training and validation sets
indices = list(range(len(input_texts)))
input_train_indices, input_val_indices, target_train_indices, target_val_indices = train_test_split(
    indices,
    indices,
    test_size=0.2,
    random_state=42
)

input_train = tf.gather(input_encodings["input_ids"], input_train_indices)
input_val = tf.gather(input_encodings["input_ids"], input_val_indices)
target_train = tf.gather(target_encodings["input_ids"], target_train_indices)
target_val = tf.gather(target_encodings["input_ids"], target_val_indices)

# Define training parameters
batch_size = 5
epochs = 200
learning_rate = 1e-5

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Training loop
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(0, len(input_train), batch_size):
        input_batch = input_train[i : i + batch_size]
        target_batch = target_train[i : i + batch_size]

        with tf.GradientTape() as tape:
            outputs = model(input_batch, labels=target_batch)
            logits = outputs.logits
            loss = loss_fn(target_batch, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss.numpy()

    # Calculate validation loss
    val_loss = 0.0
    for i in range(0, len(input_val), batch_size):
        input_batch = input_val[i : i + batch_size]
        target_batch = target_val[i : i + batch_size]

        outputs = model(input_batch, labels=target_batch)
        logits = outputs.logits
        val_loss += loss_fn(target_batch, logits).numpy()

    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(input_train)}, Validation Loss: {val_loss/len(input_val)}")

# Save the fine-tuned model
model.save_pretrained("fine-tuned-bart-title-generator")

# Generate titles for new text
input_text = ""
input_encoding = tokenizer(input_text, return_tensors="tf", padding=True, truncation=True)
generated_ids = model.generate(input_encoding["input_ids"])
generated_title = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated Title: {generated_title}")