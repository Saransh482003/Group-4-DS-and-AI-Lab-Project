from collections import OrderedDict


class TtsPhraseCache:
	def __init__(self, maxsize=128):
		self.maxsize = max(0, int(maxsize))
		self._cache = OrderedDict()

	def get(self, key):
		if self.maxsize <= 0:
			return None
		value = self._cache.get(key)
		if value is not None:
			self._cache.move_to_end(key)
		return value

	def put(self, key, value):
		if self.maxsize <= 0:
			return
		self._cache[key] = value
		self._cache.move_to_end(key)
		while len(self._cache) > self.maxsize:
			self._cache.popitem(last=False)