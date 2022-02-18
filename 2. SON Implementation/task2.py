

from pyspark import SparkContext
import time
import sys
import math
import itertools

class Bitmap:
    Bitmap = 0
    def __init__(self):
        self.Bitmap = pow(2,10000000)
    
    def sets(self, bucketNo):
        pointer = 1
        pointer = pointer << bucketNo
        self.Bitmap = self.Bitmap | pointer
    
    def contains(self, bucketNo):
        pointer = 1
        pointer = pointer << bucketNo
        checkResult = self.Bitmap & pointer
        if checkResult > 0:
            return True
        else:
            return False

class Bucket:
    Buckets = []
    totalBuckets = 0
    threshold = 0
    bitmap = Bitmap()
    def __init__(self, _totalBuckets, _threshold):
        self.totalBuckets = _totalBuckets
        self.threshold = _threshold
        self.Buckets = [0 for _ in range(self.totalBuckets)]
    
    def doHash(self, key):
        index = hash(key) % self.totalBuckets
        if self.Buckets[index] <= self.threshold:
            self.Buckets[index] = self.Buckets[index] + 1

    def saveBitmap(self):
        _bitmap = Bitmap()
        for i in range(self.totalBuckets):
            if self.Buckets[i] >= self.threshold:
                _bitmap.sets(i)
        self.bitmap = _bitmap
    
    def bitmapChecks(self, key):
        index = hash(key) % self.totalBuckets
        return self.bitmap.contains(index)

def GSS(baskets,SUPPORT):
    hashmap = {}
    for basket in baskets:
        for element in basket:
            if element not in hashmap:
                hashmap[element] = 1
            else:
                hashmap[element] = hashmap[element] + 1
    deleteKeyList = []
    for item in hashmap:
        if hashmap[item] < SUPPORT:
            deleteKeyList.append(item)
    for item in deleteKeyList:
        hashmap.pop(item)
    singletons = []
    for item in hashmap:
        singletons.append([item])
    return singletons

def isAllSubsetsFrequent(frequents, candidate):
    for index in range(len(candidate)):
        _candidate = candidate[:]
        _candidate.pop(index)
        if _candidate not in frequents:
            return False
    return True

def GPC(frequents):
    possibleCandidates = []
    for ione in range(0, len(frequents) - 1):
        itemOne = frequents[ione]
        for itwo in range(ione + 1, len(frequents)): # Jiabin Change 1
            itemTwo = frequents[itwo]
            if itemOne != itemTwo and itemOne[:-1] == itemTwo[:-1]:
                candidate = itemOne + itemTwo[-1:]
                candidate.sort()
                possibleCandidates.append(candidate)
    
    deleteIndex = []
    for index in range(len(possibleCandidates)):
        candidate = possibleCandidates[index]
        if isAllSubsetsFrequent(frequents, candidate) == False:
            deleteIndex.append(index)
    
    deleteIndex.sort(reverse=True)
    for index in deleteIndex:
        possibleCandidates.pop(index)
    return possibleCandidates

def GRC(possibleCandidates, baskets, capacity, threshold):
    if capacity > 2:
        return possibleCandidates
    bucket = Bucket(4, threshold) 
    for basket in baskets:
        candidates = list(itertools.combinations(basket, capacity))
        for candidate in candidates:
            candidate = list(candidate)
            candidate.sort()
            candidate = tuple(candidate)
            bucket.doHash(candidate)
    
    bucket.saveBitmap()
    realCandidates = []
    for index in range(len(possibleCandidates)):
        candidate = possibleCandidates[index]
        candidate.sort()
        if bucket.bitmapChecks(tuple(candidate)):
            realCandidates.append(candidate)
    
    return realCandidates

def isInBaskets(candidate, baskets, support):
    candidate = set(candidate)
    count = 0
    for basket in baskets:
        if candidate.issubset(basket):
            count = count + 1
    if count >= support:
        return True
    return False

def countInBastkets(candidate, baskets):
    candidate = set(candidate)
    count = 0
    for basket in baskets:
        if candidate.issubset(basket):
            count = count + 1
    return count

def Count(possibleCandidates, baskets, support):
    realFrequents = []
    for candidate in possibleCandidates:
        if isInBaskets(candidate, baskets, support):
            realFrequents.append(candidate)
    return realFrequents

def JPCY(baskets, SUPPORT, task):
    singletons = GSS(baskets,SUPPORT)
    permission = False
    if len(singletons):
        permission = True
    lastFrequents = singletons
    capacity = 2
    result = []
    result.append(singletons)
    while permission:
        possibleCandidates = GPC(lastFrequents)
        if len(possibleCandidates) == 0:
            break
        realCandidates = GRC(possibleCandidates, baskets, capacity, SUPPORT)
        if len(realCandidates) == 0:
            break
        newFrequents = Count(realCandidates, baskets, SUPPORT)
        if len(newFrequents) == 0:
            break
        capacity = capacity + 1
        lastFrequents = newFrequents
        result.append(newFrequents)
    return result

def SPC(baskets, candidates):
    result = []
    for candidate in candidates:
        result.append((tuple(candidate), countInBastkets(candidate, baskets)))
    return result


def GS(possibleCandidates, title):
    text = title + "\n"
    largest = len(possibleCandidates[len(possibleCandidates) - 1])
    alldata = {}
    for i in range(1,largest + 1):
        alldata[i] = []
    for item in possibleCandidates:
        item = list(item)
        for i in range(0,len(item)):
            item[i] = "'" + item[i] + "'"
        item.sort()
        alldata[len(item)].append(item)
    
    for index in range(1,largest + 1):
        seq = alldata[index]
        for i in range(len(seq)):
            text = text + "(" + ", ".join(seq[i]) + ")"
            if i < len(seq) - 1:
                text = text + ","
        if index < largest:
            text = text + "\n\n"
    return text


if __name__ == "__main__":
    time_start=time.time()    
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("ERROR")

    shift = 0
    Th = int(sys.argv[1])
    S = int(sys.argv[2])
    IF = sys.argv[3]
    OF = sys.argv[4]

    RDD_text = sc.textFile(IF)
    Head = RDD_text.first()
    B = RDD_text.filter(lambda _: _ != Head)
    B = B.map(lambda line: (line.split(',')[abs(0 - shift)], [line.split(',')[abs(1 - shift)]]))
    B = B.reduceByKey(lambda a,b : a + b)
    B = B.mapValues(lambda values: set(values))
    B = B.filter(lambda x: len(x[1]) > Th).map(lambda x: x[1])
    PNo = B.getNumPartitions()
    _PCs = B.mapPartitions(lambda chunk: JPCY(list(chunk), math.ceil(S / PNo), "1")).collect()
    PCs = []
    for group in _PCs:
        for item in group:
            if item not in PCs:
                item.sort()
                PCs.append(item)
    F = B.mapPartitions(lambda chunk: SPC(list(chunk),PCs))
    F = F.reduceByKey(lambda a,b: a + b)
    F = F.filter(lambda pair: pair[1] >= S)
    F = F.map(lambda x: x[0])
    F = F.sortBy(lambda a: (len(a), list(a))).collect()
    PCs.sort(key = lambda a: len(a))


    Fr = GS(F,"Frequent Itemsets:")
    Cand = GS(PCs,"Candidates:")

    with open(OF, 'w') as f:
        f.write(Cand + "\n\n" + Fr)

    time_end=time.time()
    print('Duration: ',time_end-time_start,'s')