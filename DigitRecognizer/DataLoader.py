
class DataLoader(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert(len(self.data) == len(self.labels))
        self.length = len(self.data)
        self.index = 0

    def hasNextBatch(self):
        return self.index < self.length

    def reset(self):
        self.index = 0

    def getNextBatch(self, batch_size):
        if (self.index + batch_size < self.length):
            self.index += batch_size
            return self.data[self.index:self.index + batch_size], self.labels[self.index:self.index + batch_size]
        else:
            self.index = self.length
            return self.data[self.index:self.length], self.labels[self.index:self.length]