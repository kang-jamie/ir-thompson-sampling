import numpy as np
from scipy.stats import beta
import scipy.integrate as integrate

class IR_ThompsonSampling(object):
	def __init__(self, true_p, out_opt):
		self.true_p = true_p
		self.out_opt = out_opt
		self.n_agents = len(out_opt)

		self.n_success = np.ones(2)
		self.n_fail = np.ones(2)
		self.n_out = 0

		self.welfare = 0


	def learn(self):
		for t in range(self.n_agents):
			if t % 100 == 0: 
				print("====="+ str(t) + "-th trial " +"=====")
			# p_samples = np.random.beta(n_success, n_fail)
			expected_max_reward = integrate.quad(lambda x: 
				1 - beta.cdf(x,self.n_success[0],self.n_fail[0])*beta.cdf(x,self.n_success[1],self.n_fail[1]), 
				0, 1)[0]

			if expected_max_reward <= self.out_opt[t]:
				# print("outside option -- " + str(self.out_opt[t]) + " > " + str(expected_max_reward))
				self.n_out += 1
				self.welfare += self.out_opt[t]

			else:
				p_samples = [beta.rvs(s,f) for s,f in zip(self.n_success,self.n_fail)]
				best_arm = np.argmax(p_samples)
				best_sample = np.max(p_samples)
				best_p = self.true_p[best_arm]
				# print("best: " + str(best_arm) + " with sample: " + str(best_sample) + " and true: " + str(best_p) + " w/ out: " + str(out_opt[t]))
				if np.random.binomial(1, best_p) == 1:
					self.n_success[best_arm] += 1
				else:
					self.n_fail[best_arm] += 1

				self.welfare += best_p

	def compute_loss(self):
		true_best_p = np.max(self.true_p)
		# return self.out_opt
		true_best_rewards = np.maximum(self.out_opt, true_best_p)
		return np.sum(true_best_rewards) - self.welfare

#   def pick_arm(self):
#       p_samples = np.random.beta(self.n_success, self.n_fail)
#       np.argmax(p_samples)

#   def pick_arm(self):

#   def update(self):

