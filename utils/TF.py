from Transform import *
import numpy as np
from scipy.spatial.transform import Rotation as R

class TFNode:
    def __init__(self, parent=None, translation=None, rotation=None):
        self.parent = parent
        self.translation = np.array(translation) if translation is not None else np.zeros(3)
        self.rotation = R.from_quat(rotation) if rotation is not None else R.identity()
        self.children = {}

class TFTree:
    def __init__(self):
        self.nodes = {'world': TFNode()}  # 根节点为 "world"

    def add_node(self, child_name, parent_name, translation, rotation):
        """
        添加子节点以及相对于父节点的变换 (translation, rotation)。
        translation: (x, y, z)
        rotation: 四元数 (x, y, z, w)
        """
        if parent_name not in self.nodes:
            raise ValueError(f"Parent node {parent_name} not found.")
        parent_node = self.nodes[parent_name]
        child_node = TFNode(parent=parent_node, translation=translation, rotation=rotation)
        self.nodes[child_name] = child_node
        parent_node.children[child_name] = child_node
        
    def delete_node(self,node_name):
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found.")
        node = self.nodes[node_name]
        parent = node.parent
        if parent is not None:
            del parent.children[node_name]
        for child_name in node.children:
            child = node.children[child_name]
            child.parent = parent
            parent.children[child_name] = child
        del self.nodes[node_name]
        return
    
    def get_transform(self, from_node, to_node):
        """
        获取从 `from_node` 到 `to_node` 的变换矩阵 (4x4)。
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("One or both of the nodes are not found in the TF tree.")
        
        # 获取 from_node 到世界的变换矩阵
        transform_from_to_world = self._get_transform_to_world(from_node)
        
        # 获取 to_node 到世界的变换矩阵
        transform_to_to_world = self._get_transform_to_world(to_node)
        
        # from -> to 的变换矩阵等于 to 的世界变换的逆乘以 from 的世界变换
        return np.linalg.inv(transform_to_to_world) @ transform_from_to_world

    def _get_transform_to_world(self, node_name):
        """
        获取从 `node_name` 到世界坐标系的4x4变换矩阵。
        """
        node = self.nodes[node_name]
        T = np.eye(4)
        
        while node is not None:
            # 将平移转换为4x4矩阵
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = node.translation
            
            # 将旋转转换为4x4矩阵
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = node.rotation.as_matrix()
            
            # 当前节点的变换矩阵 = 平移矩阵 * 旋转矩阵
            current_transform = translation_matrix @ rotation_matrix
            
            # 累积当前节点的变换
            T = current_transform @ T
            
            # 上升到父节点
            node = node.parent
        
        return T

    def transform_pose(self, pos,ori, frame_node, reference_node):
        """
        将 `frame_node` 中的pose转换到 `reference_node` 下的坐标系中。
        pos: (x, y, z) 点的坐标
        ori: (x,y,z,w) 位姿的四元数
        frame_node: 点所在的节点
        reference_node: 需要转换到的参考节点
        """
        if frame_node not in self.nodes or reference_node not in self.nodes:
            raise ValueError("Invalid node names.")
        
        # 从 point_node 到 reference_node 的变换矩阵
        T_frame2refer = self.get_transform(frame_node, reference_node)
        
        # 将点转换为齐次坐标
        T_target2frame = np.eye(4)
        T_target2frame[:3, 3] = np.array(pos)
        T_target2frame[:3, :3] = R.from_quat(ori).as_matrix()

        # 将点从 frame_node 转换到 reference_node
        T_target2refer = T_frame2refer @ T_target2frame
 
        # 返回转换后的 (x, y, z) 和 (x, y, z, w)
        return self.transform_to_pose(T_target2refer)
    def update_node(self,node_name,translation,rotation):
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found.")
        node = self.nodes[node_name]
        node.translation = translation
        node.rotation = R.from_quat(rotation)
        return
    
    def transform_to_pose(self,T):
        """
        将变换矩阵转换为位姿
        T: 变换矩阵
        pos: 位姿[x,y,z]
        ori: 位姿[x,y,z,w]
        """
        pos = T[:3,3]
        ori = R.from_matrix(T[:3,:3]).as_quat()
        return pos,ori
    
    def print_tree_structure(self):
        """
        打印树的父子结构。
        """
        print("TF Tree structure:")
        def print_node(node, indent=""):
            print(f"{indent}{node}")
            for child in self.nodes[node].children:
                print_node(child, indent + "  ")

        print_node("world")

    def print_node_info(self, node_name):
        """
        输入节点名，打印节点的子节点名、父节点名以及其相对于父节点的变换矩阵。
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found.")
        
        node = self.nodes[node_name]
        parent_name = node.parent if node.parent is None else [k for k, v in self.nodes.items() if v == node.parent][0]
        children_names = list(node.children.keys())
        translation = node.translation
        rotation = node.rotation.as_quat()

        print(f"Node: {node_name}")
        print(f"Parent: {parent_name}")
        print(f"Children: {children_names}")
        print(f"Transform to parent:\n{translation}\n{rotation}")
        
# 测试代码
if __name__ == "__main__":
    tf_tree = TFTree()
    
    # 添加子节点相对于父节点的变换
    tf_tree.add_node('child1', 'world', [1, 0, 0], [0, 0, 0, 1])
    tf_tree.add_node('child2', 'child1', [0, 1, 0], [0, 0, 0, 1])
    
    # 获取 child1 相对于 world 的变换
    print("Transform from world to child1:\n", tf_tree.get_transform('world', 'child1'))
    
    # 获取 child2 相对于 world 的变换
    print("Transform from world to child2:\n", tf_tree.get_transform('world', 'child2'))
    
    # 将 child2 中的一个点 (1, 0, 0) 转换到 world 坐标系
    point_in_world = tf_tree.transform_point([1, 0, 0], 'child2', 'world')
    print("Point in child2 transformed to world:", point_in_world)
