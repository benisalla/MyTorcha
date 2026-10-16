from MyTorcha.Node import Node


def sum_nodes(nodes):
    tot = Node(0.0)
    for node in nodes:
        tot = tot + node
    tot._nm = "S"
    return tot
