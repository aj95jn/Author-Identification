"""
Ankit Jain

aj9761

Lab 2
"""

# 1: for Arthur,
# 0: Herman
import math
import numpy
import sys
featureList = ["AvgWordsInSentence", "SemiColen", "ResponseWords", "QuotationSymbols", "SherlockNames"]

"""
Class to build tree for decision tree
"""
class Node:
    __slots__ = 'featureName', 'leftAttribute', 'rightAttribute'  # slots or constants that will be nodes for tre
    def __init__(self, rootNode, leftNode, rightNode):
        self.featureName = rootNode # root of tree
        self.leftAttribute = leftNode # left chid of tree
        self.rightAttribute = rightNode # right tree of tree



"""
This function and on below is to find the new reduced matrix as we move down the tree. This particular function will
will go for the left subtree where the value should be 1. If it is, leave that entry as we already know we are at left
subtree and append rest of the attributte values in the new matrix
"""
def reducedMatrixleft(tempfmat, column):
    tempnewleftmat = []  # temperoroy left matrix that keeps changing with every left parse of tree
    for i in range(len(tempfmat)):
        for j in range(len(tempfmat[0])-1):
            if j == column: # if the column is the best attribute, chack for 1 or 0
                if tempfmat[i][j] == 1:
                    x = tempfmat[i][:column] # if 1, leaving that, consider all otehers in a seperate matrix
                    y = tempfmat[i][column + 1:]
                    tempnewleftmat.append(x+y)
                    break
    return tempnewleftmat


"""
Matches the riht subtree of the root. If value is 0, that value is left from the best selected attribute and the rest 
have to be taken into account
"""
def reducedMatrixright(tempfmat, column):
    tempnewrightmat = []
    for i in range(len(tempfmat)):
        for j in range(len(tempfmat[0])-1):
            if j == column:
                if tempfmat[i][j] == 0: # if value is 0, leave that from the best attribute, consider the rest
                    x = tempfmat[i][:column]
                    y = tempfmat[i][column + 1:]
                    tempnewrightmat.append(x + y)
                    break
    return tempnewrightmat


"""
Function of decision tree. It will build the entire tree based on 2 base conditions, if length of matrix passed is 0, it
should return the decision on that level or if entropy is 0, the classification is done and we can knw which author 
was it at that point 
"""
def decision_tree(ffmat, E):
    if E == 0:
        return
    elif len(ffmat) == 0:
        return
    BestAtt, entropyLeft, entropyRight = selectBestAttribute(ffmat) # the best attribute column index is returned along with
    # left entropy and right entropy as the left entropy will be the Einit value for the next level.
    parentfeature = Node(featureList[BestAtt], None, None)
    tempFeature = ffmat[:] # saving values in another matrix as this matrix will change
    parentfeature.leftAttribute = decision_tree(reducedMatrixleft(tempFeature, BestAtt), entropyLeft)
    parentfeature.rightAttribute = decision_tree(reducedMatrixright(tempFeature, BestAtt), entropyRight)
    return parentfeature # returns the node that is formed

"""
Function that will select the best attribute at every parse of the decision tree. It returns the index of the column
"""
def selectBestAttribute(fmat):
    Gainlist = [] # accumulate all gains first of all columns in this list and then take maximum amongst them
    Eleft = [] # the entropy of the left node
    Eright =[] # entropy of the right nodes
    left_positive = 0
    left_negative = 0
    right_positive = 0
    right_negative = 0
    positive = 0
    negative = 0
    for i in range(len(fmat[0])-1): # for the number of coulumns in the matrix
        for j in range(len(fmat)): # for the entirity of the length of the feature matrix
            # we are checking and calculating entropy based on number of positives and negatives.
            if fmat[j][i] == 1 and fmat[j][-1] == 1: # if the particular attribute values and last deciding are 1
                # it is on left yes, yes
                left_positive = left_positive + 1
                positive += 1 # take count of positives
            if fmat[j][i] == 1 and fmat[j][-1] == 0: # true, false on left
                left_negative += 1
                negative += 1
            if fmat[j][i] == 0 and fmat[j][-1] == 0: # false, false on right
                right_negative += 1
                negative += 1
            if fmat[j][i] == 0 and fmat[j][-1] == 1: # false, true on right
                right_positive += 1
                positive += 1


        Einit = entropy(positive, negative, (positive + negative)) # entropy calculation with positives , negatives and total count passed
        eleft = entropy(left_positive, left_negative, left_positive + left_negative)
        eright = entropy(right_positive, right_negative, right_positive + right_negative)
        Eleft.append(eleft)
        Eright.append(eright)
        Gain = gain(Einit, eleft , eright, (left_positive + left_negative), (right_negative + right_positive), (negative + positive))

        Gainlist.append(Gain)
        left_positive = 0
        left_negative = 0
        right_positive = 0
        right_negative = 0
        positive = 0
        negative = 0

    BestAttributeIndex = Gainlist.index(max(Gainlist)) # taking the index of best attribute
    print("The best attribute is column no: ", BestAttributeIndex)
    return BestAttributeIndex, Eleft[BestAttributeIndex], Eright[BestAttributeIndex]

"""
Calculation of entropy
"""


def entropy(positives, negatives, total):
    if total != 0:
        positive_prob = (positives / total)
        negative_prob = (negatives / total)
    else:
        positive_prob = 0
        negative_prob = 0

    if positive_prob == 0 or negative_prob == 0:
        return 0
    else:
        Entropy = (-positive_prob * math.log(positive_prob, 2)) - (negative_prob * math.log(negative_prob, 2))
    return Entropy


"""
Calculation for gain
"""
def gain(Einit, Ent_left, Ent_right, left_total, right_total, total):
    gain = 0
    if total != 0:
        gain = Einit - (((left_total / total) * Ent_left) + ((right_total / total) * Ent_right))
    return gain


"""
Function that trains the data for logistic regression and update weights accordingly
"""
def WeightCalculation(ffmat, initweightlist, alpha, iterations):
    tempweightlist = []
    for i in range(0, iterations): # for the number of examples considered
        sigfunc = 1/ (1 + math.exp(-(numpy.dot(ffmat[i][0:-1], initweightlist)))) # taking the sigmoid of the weights and the feature matrix
        for j, k in zip(ffmat[i], initweightlist):
            tempweightlist.append(k + alpha*(ffmat[-1][0] - sigfunc) * sigfunc * (1 - sigfunc) * j) # updating weights
        initweightlist = tempweightlist
        tempweightlist = []
        alpha = alpha - math.sqrt(alpha)
    return initweightlist


"""
Below are the features on which the clasification will be done
"""
def avgWordCountInSentence(para):
    wordcounts = []
    sentences = para.split('.')
    for sentence in sentences:
        words = sentence.split(' ')
        wordcounts.append(len(words))

    average_wordcount = sum(wordcounts) / len(sentences)
    if average_wordcount <= 21:
        return 1
    else:
        return 0


def semicolen(para):
    count = sum(line.count(';') for line in para)
    if count < 3:
        return 1
    else:
        return 0


def replyWords(para):
    if 'said' in para or 'answered' in para or 'remarked' in para or 'replied' in para:
        return 1
    else:
        return 0


def quotationSymbols(para):
    count = sum(line.count('“') or line.count('”') or line.count('"') for line in para)
    if count >= 3:
        return 1
    else:
        return 0


def famousNames(para):
    if '211' in para or 'Dr.' in para or 'Watson' in para or 'Dr. Watson' in para or 'Baker' in para or 'Holmes' in para or 'Sherlock' in para or 'Moriarty' in para\
                or 'Hudson' in para or 'John' in para or 'Mycroft' in para or 'Lestrade' in para or 'Mary' in para or 'Irene' in para:
        return 1
    else:
        return 0


"""
Function that will check for appropriate feature in the passed paragraphs
"""
def checkFeature(para):

    a = avgWordCountInSentence(para)
    b = semicolen(para)
    c = replyWords(para)
    d = quotationSymbols(para)
    e = famousNames(para)
    featureVal = []
    featureVal.append(a)
    featureVal.append(b)
    featureVal.append(c)
    featureVal.append(d)
    featureVal.append(e)
    return featureVal


def main():
    finalFeatureMat = [] # list that holds the final weights after training
    Initialweightslist = [] # initial weights assigned as 0
    if len(sys.argv) > 0 and sys.argv[1] == 'train':
        print("AvgWordsInSentence, SemiColen, ResponseWords, QuotationSymbols, SherlockNames, Arthur/Herman")
        with open('TrainingData.txt', "r") as f:
            train_data = f.read().split('\n\n')
            for paragraphs in train_data:
                paragraph = paragraphs.split('->')
                flist = checkFeature(paragraph[0])
                finalFeatureMat.append(flist)
                if paragraph[1] == 'A':
                    flist.append(1)
                else:
                    flist.append(0)
            for sublist in finalFeatureMat:
                print(sublist)
        for weight in range(0, len(finalFeatureMat[0])-1):
            Initialweightslist.append(1)

        finalweightlist = WeightCalculation(finalFeatureMat, Initialweightslist, 1, 70)
        print(finalweightlist)
        f = open("weight.txt", "w")
        for i in finalweightlist:
            f.write(str(i) + ", ")

    elif len(sys.argv) > 2:
        testfile = sys.argv[2]
        weights = []
        t = open("weight.txt", "r")
        finalweights = t.read()
        fw1 = finalweights.strip().split(',')
        fw2 = fw1[:-1]
        for w in fw2:
            weights.append(float(w))
        print(weights)
        file = open(testfile, "r")
        predictdata = file.read()
        predictList = checkFeature(predictdata)
        print(predictList)
        val = (1 / (1 + math.exp(-(numpy.dot(predictList, weights)))))
        if val > 0.5:
            print("the author is Arthur")
        else:
            print("The author is Herman")


    """
    Code no complete, but runs for best attribute.
    """
    # if choice2 == 'train' or choice2 == 'Train' or choice2 == 'TRAIN':
    #     feature = decision_tree(finalFeatureMat, 1)  # to start of with bad or no classification, and subsequently
                                                     #  reduce until 0 or close to 0.
        # print(feature)


if __name__=="__main__":
    main()