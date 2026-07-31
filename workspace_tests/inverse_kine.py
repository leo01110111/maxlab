#given an end effector pose, how well can a neural network do inverse kinematics?

#labeled dataset?
import torch
import gymnasium as gym

ur_sim = gym.make("UR")
joint_dim = ur_sim.action_space()
#collect random poses

dataset = []

for i in range(1000):
    rand_action = ur_sim.action_space.sample()
    pose_left, pose_right = obtain_pose() #how do i implement this?
    dataset.append(rand_action, pose)

def obtain_pose():
    pass

class Network(torch.nn.Module):
    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(pose_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, joint_dim),
        )
    
    def forward(self, x):
        return self.model(x)



train_loader = torch.Dataloader(dataset, batch_size=10)

net = Network()
net.model.train()
loss = torch.MSELoss()
optim = torch.optim.Adam(lr=1e-3)

step = 0

for epoch in 2:
    for x, label in train_loader:
        y = net.model(x)
        loss_val = loss(x, y)
        net.zero_grad()
        net.backwards()
        optim.step()
        step += 1
        
        if step % 10 == 0:
            print("train loss", loss_val)
            print("eval begins")
            

