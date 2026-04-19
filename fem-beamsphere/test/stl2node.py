import sys
import struct


def is_binary_stl(filename):
    """检查STL文件是否为二进制格式"""
    try:
        with open(filename, 'rb') as f:
            # 读取文件头（80字节）和三角形数量（4字节）
            header = f.read(84)
            if len(header) < 84:
                return False

            # 解析三角形数量
            num_triangles = struct.unpack('<I', header[80:84])[0]

            # 计算预期的文件大小
            expected_size = 84 + num_triangles * 50

            # 获取实际文件大小
            f.seek(0, 2)  # 移动到文件末尾
            actual_size = f.tell()

            # 如果实际大小与预期大小匹配，则是二进制文件
            return actual_size == expected_size
    except:
        return False


def read_binary_stl_vertices(stl_file):
    """从二进制STL文件中读取顶点"""
    vertices = []

    with open(stl_file, 'rb') as f:
        # 跳过80字节的文件头
        f.read(80)

        # 读取三角形数量
        num_triangles = struct.unpack('<I', f.read(4))[0]

        # 读取每个三角形
        for _ in range(num_triangles):
            # 跳过法线向量（12字节）
            f.read(12)

            # 读取三个顶点（每个顶点12字节）
            for _ in range(3):
                x = struct.unpack('<f', f.read(4))[0]
                y = struct.unpack('<f', f.read(4))[0]
                z = struct.unpack('<f', f.read(4))[0]
                vertices.append((x, y, z))

            # 跳过属性字节计数（2字节）
            f.read(2)

    return vertices


def read_ascii_stl_vertices(stl_file):
    """从ASCII STL文件中读取顶点"""
    vertices = []

    with open(stl_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            if parts[0] == 'vertex':
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    vertices.append((x, y, z))
                except ValueError:
                    continue

    return vertices


def write_stl_vertices_to_node(stl_file, node_file):
    """转换STL顶点到node格式"""
    # 检查文件格式
    if is_binary_stl(stl_file):
        vertices = read_binary_stl_vertices(stl_file)
        print("检测到二进制STL格式")
    else:
        vertices = read_ascii_stl_vertices(stl_file)
        print("检测到ASCII STL格式")

    # 写入.node文件
    with open(node_file, 'w') as f:
        # 文件头
        f.write(f"{len(vertices)} 3 0 0\n")

        # 写入每个顶点
        for i, v in enumerate(vertices, 1):
            f.write(f"{i} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

    return len(vertices)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python stl2node.py <input.stl> <output.node>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        vertex_count = write_stl_vertices_to_node(input_file, output_file)
        print(f"成功转换 {input_file} 到 {output_file}")
        print(f"提取了 {vertex_count} 个顶点")
        print("注意: 输出可能包含重复顶点。使用TETGEN的-d选项去除重复。")
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        sys.exit(1)