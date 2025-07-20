import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

ACTIONS = ["C", "D"]
PAYOFFS = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1)
}
EPISODES = 1000
LEARN_INTENSITY = 0.5
MIN_MAX_HISTORY_LENGTH = (5, 25)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_length = 25
        self.net = nn.Sequential(
            nn.Linear(self.history_length * 2, 34),
            nn.ReLU(),
            nn.Linear(34, 2),
        )
    
    def forward(self, state):
        return self.net(state)

    def choose_action(self, state):
        with torch.no_grad():
            logits = self(state)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return ACTIONS[action], action



def history_to_state(history):
    x = torch.zeros(25, 2)
    for i, (a, b) in enumerate(history[-25:]):
        x[i][0] = 1 if a == "C" else 0
        x[i][1] = 1 if b == "C" else 0
    return x.flatten()



def train(agent, episodes=EPISODES, learning_rate=0.01):
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    agent.train()

    for episode in tqdm(range(episodes)):
        history = []
        total_a = 0
        total_b = 0
        turns = random.randint(MIN_MAX_HISTORY_LENGTH[0], MIN_MAX_HISTORY_LENGTH[1])

        for turn in range(turns):
            state = history_to_state(history)

            action_a, idx_a = agent.choose_action(state)
            action_b, idx_b = agent.choose_action(state)

            reward_a, reward_b = PAYOFFS[(action_a, action_b)]
            total_a += reward_a
            total_b += reward_b
            history.append((action_a, action_b))


            reward = torch.tensor([(total_a-(turn+1))*LEARN_INTENSITY], dtype=torch.float32)

            # Backprop
            logits = agent(state)
            target = logits.clone().detach()
            target[idx_a] = reward

            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 1000 == 0:
            print(f"\nEpisode {episode} | Score A: {total_a}, Score B: {total_b}, Rounds: {turns}")
    agent.eval()
    torch.save(agent.state_dict(), "prisoner_dilema_model_v2.pth")

def play(agent):
    agent.eval()

    while True:
        history = []
        you = 0
        ai = 0
        rounds = int(input("how many rounds do you want to play?(25 max): "))
        if rounds == 0:
            rounds = random.randint(5, 25)

        print("type 'C' to cooperate and 'D' to defect.\n")

        for round_num in range(rounds):
            while True:
                human_move = input(f"[Round {round_num}] Your move (C/D): ").strip().upper()
                if human_move in ACTIONS:
                    break
                print("Invalid input. Please enter C or D.")

            state = history_to_state(history)
            ai_move, _ = agent.choose_action(state)

            reward_you, reward_ai = PAYOFFS[(human_move, ai_move)]
            you += reward_you
            ai += reward_ai
            history.append((human_move, ai_move))

            print(f"  You played: {human_move}")
            print(f"  AI played:  {ai_move}")
            print(f"  Round result: You +{reward_you}, AI +{reward_ai}")
            print("")

        print("=== GAME OVER ===")
        print(f"Your total score: {you}")
        print(f"AI total score:   {ai}")
        if you > ai:
            print("🎉 You win!")
        elif you < ai:
            print("🤖 AI wins!")
        else:
            print("🤝 It's a tie!")

        if input("do you want to play again?(y/n): ") != "y":
            break
        else:
            print("\n\n")




if __name__ == "__main__":
    agent = Agent()
    if input("train or play?(t/p):") == "t":
        train(agent, episodes=10000, learning_rate=0.01)
        print("Training complete.")
        
        if input("Do you want to play now?(y/n): ") != "y":
            print("ok, load it later!")
        else:
            print("lets go!\n\n")
            play(agent)
    else:
        agent.load_state_dict(torch.load("prisoner_dilema_model_v2.pth"))
        agent.eval()
        play(agent)
