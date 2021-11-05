import torch

class Matching:
	def __init__(self, matching, trace):
		self.matching = matching
		self.trace = trace


	def detach(self):
		matching = self.matching.detach()
		trace = self.trace
		return Matching(matching, trace)
