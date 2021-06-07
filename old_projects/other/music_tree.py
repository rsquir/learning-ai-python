class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)



m_i = Node('c')
m_i_i = Node('c')
m_i.add_child(m_i_i)
m_i_i_i = Node('c')
m_i_i.add_child(m_i_i_i)
m_i_i_ii = Node('d')
m_i_i.add_child(m_i_i_ii)
m_i_i_iii = Node('e')
m_i_i.add_child(m_i_i_iii)
m_i_ii = Node('d')
m_i.add_child(m_i_ii)
m_i_ii_i = Node('c')
m_i_ii.add_child(m_i_ii_i)
m_i_ii_ii = Node('d')
m_i_ii.add_child(m_i_ii_ii)
m_i_ii_iii = Node('e')
m_i_ii.add_child(m_i_ii_iii)


for n in m_i.children:
	print n.data





value = raw_input("first note (c):\n")

print("note is: ", value)