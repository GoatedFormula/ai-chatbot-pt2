import os
import torch
from Train import BigramLanguageModel
from Train import decode
from Train import encode
from Train import device
from Train import checkpoint_file

checkpoint = torch.load(checkpoint_file, map_location=device)
chat_history = "chat_history.txt"

max_new_tokens = 200

model = BigramLanguageModel().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

conversation = ""
if os.path.exists(chat_history):
    with open(chat_history, "r", encoding="utf-8") as f:
        conversation = f.read()
    print("--- Previous Chat History Loaded ---")
    print(conversation, end="")
    print("-----------------------------------")

if __name__ == "__main__":

    while True:

        user_input = input("<user> ")

        if user_input.lower() == "exit":
            break

        conversation += f"<user> {user_input}\n"

        with open(chat_history, "a", encoding="utf-8") as f:
            f.write(f"<user> {user_input}\n")

        context = torch.tensor([encode(conversation)],dtype=torch.long,device=device)

        output = model.generate(context,max_new_tokens)

        generated = decode(output[0].tolist())

        response = generated[len(conversation):]

        if not(response.startswith("<user>")):                       #useful only if "<user>" is in the training data

            if "<user>" in response:                                 #useful only if "<user>" is in the training data
                response = response.split("<user>")[0].strip()

            print(response.lstrip())

            with open(chat_history, "a", encoding="utf-8") as f:
                f.write(response.lstrip() + "\n")

        conversation += response.lstrip() + "\n"
