# Tejas Sharma
# October 15, 2022
# Tested by running
# cat bingo_input_1.txt | python tejas_py_bingo.py
# to supply input to the program

import sys


def matchBingoBoard(bingoBoardPattern, bingoBoardMarked):
    """
    Compare the two boards and if for any pattern there's a 1 or a 4 but there isn't a 
    corresponding 1 or 4 in the marked bingo board, return null
	However, if there is a pattern match, return an ArrayList of arrays where each array
	is the (i, j) coordinate of a match location.
	This match location will be used later on to verify the last called number was a pattern match
    """
    locationsOfMatches =  []
    isMatching = True

    for i in range(5):
        for j in range(5):
            if (bingoBoardPattern[i][j] == 1 or bingoBoardPattern[i][j] == 4) and bingoBoardMarked[i][j] != 1:
                # Mismatch between pattern, return false
                isMatching = False
            elif (bingoBoardPattern[i][j] == 1 or bingoBoardPattern[i][j] == 4) and bingoBoardMarked[i][j] == 1:
                # It's a match, add it to the matched locations
                location = [i, j]
                locationsOfMatches.append(location)
            j += 1
        i += 1

    if isMatching:
        return locationsOfMatches
    else:
        return None
        
def rotateMatrix(mat) :
    """
    rotates a matrix 90 degrees clockwise
    """

    # Initialize dimensions and output matrix
    rows = len(mat)
    cols = len(mat[0])
    res = [[0] * rows for _ in range(cols)]

    # Handle rotation (ith col and (rows - i)th row switch)
    for r in range(rows):
        for c in range(cols):
            res[c][rows - 1 - r] = mat[r][c]
            c += 1
        r += 1

    return res

def checkIfBingo(bingoBoardPattern, bingoBoardMarked, calledOutNumbers, crazyBlank, bingoBoardNumbers):
    """
    Logic to check the bingo board for straight pattern
    and crazy blank patterns
    """
    result = None

    # Handle straight blank pattern 
    if not crazyBlank:
        result = matchBingoBoard(bingoBoardPattern, bingoBoardMarked)
    else :
        # Handle crazy blank pattern
        # Check initial rotation
        result = matchBingoBoard(bingoBoardPattern, bingoBoardMarked)
        if result == None:
            # Try 90 degrees rotation
            rotatedBingoBoardPattern = rotateMatrix(bingoBoardPattern)
            result = matchBingoBoard(rotatedBingoBoardPattern, bingoBoardMarked)
            if result == None:
                # Rotate once more for a 180 degrees rotation
                rotatedBingoBoardPattern = rotateMatrix(rotatedBingoBoardPattern)
                result = matchBingoBoard(rotatedBingoBoardPattern, bingoBoardMarked)
                if result == None:
                    # Rotate once more for a 270 degrees rotation
                    rotatedBingoBoardPattern = rotateMatrix(rotatedBingoBoardPattern)
                    result = matchBingoBoard(rotatedBingoBoardPattern, bingoBoardMarked)
    # No matches
    if result == None:
        return False

    # There was a pattern match
    # Now check that the last called out number was part of the matching pattern

    # Get the last called number
    lastCalledNumber = calledOutNumbers[len(calledOutNumbers) - 1]

    # Get the location on the board of the last called number
    lastCalledNumberLocation = [-1, -1]
    for i in range(5):
        for j in range(5):
            if bingoBoardNumbers[i][j] == lastCalledNumber:
                lastCalledNumberLocation = [i, j]
            j += 1
        i += 1

    finalResult = False
    # Finally check if the last called number is part of the pattern
    for res in result :
        if res[0] == lastCalledNumberLocation[0] and res[1] == lastCalledNumberLocation[1]:
            finalResult = True
    return finalResult

def main():
    
    # Initialize bingo board data structure
    bingoBoardPattern = [[0] * (5) for _ in range(5)]
    crazyBlank = False

    # Read in the bingo board pattern
    for i in range(5):
        numbs_input = sys.stdin.readline()
        numbs = numbs_input.split(" ")
        for j in range(5):
            numb = int(numbs[j])
            if (numb == 4) :
                crazyBlank = True
            bingoBoardPattern[i][j] = numb
            j += 1
        i += 1

    # Skip the empty line
    sys.stdin.readline()

    # Read in the called out numbers to a list
    calledOutNumbers =  []
    for s in sys.stdin.readline().split(" "):
        calledOutNumbers.append(int(s))

    # Skip the empty line
    sys.stdin.readline()

    # Read in the numbers on the bingo board
    bingoBoardNumbers = [[0] * 5 for _ in range(5)]
    for i in range(5):
        numbs = sys.stdin.readline().split(" ")
        for j in range(5):
            bingoBoardNumbers[i][j] = int(numbs[j])
            j += 1
        i += 1

    # Iterate through the numbers bingo board and mark each
    # number that is part of the called out numbers with a 1
    # otherwise with a 0
    bingoBoardMarked = [[0] * 5 for _ in range(5)]
    for i in range(5):
        for j in range(5):
            if bingoBoardNumbers[i][j] in calledOutNumbers:
                bingoBoardMarked[i][j] = 1
            elif i == 2 and j == 2:
                # Center is always given for free
                bingoBoardMarked[i][j] = 1
            else :
                # Mark as 0
                bingoBoardMarked[i][j] = 0
            j += 1
        i += 1

    # After parsing in all the data and marking the board with 1s and 0s
    # call the helper checkIfBingo method to handle the checking for validity
    checkBingo = checkIfBingo(bingoBoardPattern, bingoBoardMarked, calledOutNumbers, crazyBlank, bingoBoardNumbers)
    
    if checkBingo:
        print("VALID BINGO")
    else :
        print("NO BINGO")

if __name__=="__main__":
    main()