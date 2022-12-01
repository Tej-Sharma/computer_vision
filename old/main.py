import numpy as np
import cv2 as cv

# Enter your code here. Read input from STDIN. Print output to STDOUT
from ast import literal_eval


class TreeNode(object):
    def __init__(self, val=None, left=None, right=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent


def parse_input(input_text):
    """
    Parse the input into a list of tuples containing nodes
    """
    result = []
    # Hashmap to look for duplicate pairs or more than two children
    h = {}
    try:
        pairs = input_text.split(' ')
        for pair in pairs:
            # Parse out pair to convert to a tuple
            pair = pair.replace('(', '')
            pair = pair.replace(')', '')
            split_pair = pair.split(',')
            parent = split_pair[0]
            child = split_pair[1]
            # Add to hashmap to keep track of parent/children
            if parent in h:
                if child in h[parent]:
                    # Duplicate pair
                    print('E2')
                    return None
                h[parent].append(child)
                if len(h[parent]) > 2:
                    # More than 2 children
                    print('E3')
                    return None
            else:
                h[parent] = [child]
            result.append((parent, child))
    except Exception as e:
        # If there is any formatting problem, it's an invalid input
        print('E1')
        return None
    return result


def add_pair_to_tree(curr_node, parent, child):
    if not curr_node:
        return None
    if curr_node.val == parent:
        # Found parent, add child
        if curr_node.left:
            curr_node.right = TreeNode(val=child, parent=curr_node)
            return curr_node
        else:
            curr_node.left = TreeNode(val=child, parent=curr_node)
            return curr_node
    elif curr_node.val == child:
        # Found child, need to add a parent to this child
        if curr_node.parent:
            # Already has a parent
            print('E4')
            return None
        else:
            # Create a parent for this node
            new_parent = TreeNode(val=parent, left=curr_node)
            curr_node.parent = new_parent
            return curr_node.parent
    else:
        if curr_node.left:
            add_pair_to_tree(curr_node.left, parent, child)
        if curr_node.right:
            add_pair_to_tree(curr_node.right, parent, child)
        if curr_node.parent:
            add_pair_to_tree(curr_node.parent, parent, child)


def create_tree_from_pairs(pairs):
    # points to the current root
    root = None
    # points to the current node
    curr = None
    # Initialize the tree with a root node with the child as a left child
    first_pair = pairs[0]
    root = TreeNode(val=first_pair[0])
    left_child = TreeNode(val=first_pair[1])
    root.left = left_child
    left_child.parent = root

    # Add all the other pairs
    for i in range(1, len(pairs)):
        add_pair_to_tree(root, pairs[i][0], pairs[i][1])


def main():
    input_text = '(A,B) (B,D) (D,E) (A,C) (C,F) (E,G)'
    pairs = parse_input(input_text)
    if pairs:
        create_tree_from_pairs(pairs)


main()