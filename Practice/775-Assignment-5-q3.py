class Employee(object):
    def __init__(self, name, conviviality):
        self.name = name
        self.conviviality = conviviality
        self.left_child = None
        self.right_sibling = None


def add_children(parent, children):
    """Connect children using left-child, right-sibling representation."""
    if not children:
        return
    parent.left_child = children[0]
    for i in range(len(children) - 1):
        children[i].right_sibling = children[i + 1]


def iter_children(node):
    child = node.left_child
    while child is not None:
        yield child
        child = child.right_sibling


def best_party(node):
    """
    Returns two states for each subtree rooted at node:
    1. include_state: best score/list when node attends.
    2. exclude_state: best score/list when node does not attend.
    """
    if node is None:
        return (0, []), (0, [])

    include_score = node.conviviality
    include_list = [node.name]
    exclude_score = 0
    exclude_list = []

    for child in iter_children(node):
        child_include, child_exclude = best_party(child)

        include_score += child_exclude[0]
        include_list += child_exclude[1]

        if child_include[0] > child_exclude[0]:
            exclude_score += child_include[0]
            exclude_list += child_include[1]
        else:
            exclude_score += child_exclude[0]
            exclude_list += child_exclude[1]

    return (include_score, include_list), (exclude_score, exclude_list)


def make_guest_list(president):
    include_state, exclude_state = best_party(president)
    if include_state[0] > exclude_state[0]:
        return include_state
    return exclude_state


# Example company hierarchy
if __name__ == '__main__':
    president = Employee('President', 10)
    vp_sales = Employee('VP Sales', 6)
    vp_engineering = Employee('VP Engineering', 7)
    sales_manager = Employee('Sales Manager', 8)
    account_exec = Employee('Account Executive', 5)
    eng_manager = Employee('Engineering Manager', 9)
    developer = Employee('Developer', 4)

    add_children(president, [vp_sales, vp_engineering])
    add_children(vp_sales, [sales_manager, account_exec])
    add_children(vp_engineering, [eng_manager])
    add_children(eng_manager, [developer])

    score, guests = make_guest_list(president)
    print('Maximum conviviality:', score)
    print('Guest list:', guests)
    print('Running time: O(n), because each employee is processed once.')
