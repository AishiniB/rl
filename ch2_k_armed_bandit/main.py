import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns

np.random.seed(42)

class KArmedBandit:
    def __init__(self,k=10,epsilon=0.1,verbose=True):
        self.k = k
        self.epsilon = epsilon
        self.verbose = verbose
        self._reward = Reward(k=k)
        self._action = Action(epsilon=epsilon)
        self._estimate = Estimate(k=k)
        
    def run_experiment(self, time_steps=100):
        #Keep track of stuff
        self._reward.print_testbed()
        self.times_optimal_action_chosen = 0
        self.optimality_rate_list = []
        self.total_reward = 0
        self.average_reward_list = []
        
        for time_step in range(time_steps):
            if self.verbose:
                print(f"Time step {time_step}")
            self.step(time_step)
            
        self.plot_average_reward()
        self.plot_optimality_rate()
            
        # self._estimate.print_estimates()
    
    def step(self,time_step):
        is_optimal_action = self._action.perform_action()
        chosen_action_idx, reward = self._reward.get_reward(is_optimal_action)
        current_estimates = self._estimate.update(chosen_action_idx,reward,1/(time_step+1))        
        
        if is_optimal_action:
            self.times_optimal_action_chosen += 1
        self.optimality_rate_list.append(self.times_optimal_action_chosen/(time_step+1))
        
        self.total_reward += reward
        self.average_reward_list.append(self.total_reward/(time_step+1))
        
        is_optimal_dict = {True:"Optimal",False:"Suboptimal"}
        if self.verbose:
            print(f"{is_optimal_dict[is_optimal_action]} Action chosen is {chosen_action_idx+1} with a reward of {reward} and current_estimates are {current_estimates}")
            
    def plot_average_reward(self):
        plt.plot(self.average_reward_list)
        plt.title("Average Reward Over Time")
        plt.xlabel("Time")
        plt.ylabel("Average Reward")
        plt.yticks(np.arange(-2,2,0.1))
        plt.show()
        
    def plot_optimality_rate(self):
        plt.plot(self.optimality_rate_list)
        plt.title("Optimality Rate")
        plt.xlabel("Time")
        plt.ylabel("Optimal Choice Rate")
        plt.yticks(np.arange(0,1,0.1))
        plt.show()
        
class Estimate:
    def __init__(self,k=10,alpha=0.1,default_value=0):
        self.k = k
        self.alpha = alpha
        self.Qt = [[default_value for _ in range(k)]] #This is the action value estimate at a time t, for Qt[0][3] it's Action value estimate at t=0 for a=3
        
    def update(self, chosen_action, obtained_reward, alpha=None):
        #Here time_step is before the update
        if alpha is None:
            alpha = self.alpha
        current_estimates = copy.deepcopy(self.Qt[-1])
        
        old_value_estimate = current_estimates[chosen_action]
        
        #Gradient ascent with sample averaging
        current_estimates[chosen_action] = old_value_estimate + alpha*(obtained_reward-old_value_estimate)
        
        self.Qt.append(current_estimates)
        
        return current_estimates
        
    def print_estimates(self):
        for time_step,l in enumerate(self.Qt):
            print(f"Time {time_step}: {l}")
        
class Reward:
    def __init__(self,k=10):
        self.k = k
        
        self.generate_testbed()
        
    def generate_testbed(self,mean=0,var=1):
        #Each test bed will have k actions with each reward having a mean sampled normally and variance of 1
        self.testbed = [(mean + var*np.random.standard_normal(),1) for _ in range(self.k)]
        
    def get_reward(self,is_optimal_action):
        if is_optimal_action:
            return self.get_optimal_reward()
        else:
            return self.get_suboptimal_reward()
        
    def _get_optimal_reward_idx(self):
        return np.argmax(np.array([item[0] for item in self.testbed]))
    
    def get_optimal_reward(self):
        optimal_bandit_mean = -1000
        optimal_bandit_var = 0
        optimal_bandit_idx = -1
        
        for idx,bandit_distribution in enumerate(self.testbed):
            if optimal_bandit_mean < bandit_distribution[0]:
                optimal_bandit_idx = idx
                optimal_bandit_mean = bandit_distribution[0]
                optimal_bandit_var = bandit_distribution[1]

        return (optimal_bandit_idx, optimal_bandit_mean + optimal_bandit_var*np.random.standard_normal())
    
    def get_suboptimal_reward(self):
        chosen_action_idx = np.random.choice(np.array([bandit for bandit in list(range(0,self.k,1)) if bandit != self._get_optimal_reward_idx()]))
        
        chosen_action_mean, chosen_action_var = self.testbed[chosen_action_idx]
        
        return (chosen_action_idx, chosen_action_mean + chosen_action_var*np.random.standard_normal())
    
    def print_testbed(self,top_view=True):
        density = 100
        x = np.array([])
        y = np.array([])
        z = np.array([])
        
        for bandit_no in range(self.k):
            mean, variance = self.testbed[bandit_no]
            #Here the mean is the q*(a) value for this stationary series
            print(f"For Bandit {bandit_no+1}: Mean = {mean:.4f} and Variance = {variance}") 
            
            #Some numbers for creating the plot
            x_axis = np.array([bandit_no + 1 for _ in range(10*density)])
            y_axis = np.arange(-5,5,1/density)
            z_axis = np.exp(-np.square(y_axis-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
            
            x = np.concatenate((x,x_axis),axis=None)
            y = np.concatenate((y,y_axis),axis=None)
            z = np.concatenate((z,z_axis),axis=None)                
            
        fig = plt.figure(figsize=(10,10))
        
        display_values = {"Bandit Number":x,
                          "Reward Values":y,
                          "Probability":z}
        
        if top_view:
            sns.scatterplot(data=display_values, x="Bandit Number", y="Reward Values", hue="Probability", size="Probability")
        else:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
            ax.set_xlabel('Bandit Number')
            ax.set_ylabel('Reward Values')
            ax.set_zlabel('Probability')

            # Adding a colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Probabilities')

        plt.title('K-Armed Testbed')
        plt.show()
        
        print(f"Best Bandit is {self._get_optimal_reward_idx()+1}")

class Action:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        
    def perform_action(self):
        #This action tells whether to perform the optimal (1) or suboptimal (0) action
        return np.random.random() > self.epsilon
        
if __name__ == "__main__":
    exp = KArmedBandit(k=10,epsilon=0.15)
    exp.run_experiment(time_steps=2000)