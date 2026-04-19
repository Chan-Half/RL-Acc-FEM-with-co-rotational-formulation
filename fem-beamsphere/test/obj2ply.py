import sys


def obj_to_ply(obj_file, ply_file):
    """Convert OBJ file to PLY format"""
    vertices = []
    faces = []
    vertex_count = 0
    face_count = 0

    # 读取OBJ文件
    with open(obj_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':  # 顶点
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    vertex_count += 1

            elif parts[0] == 'f':  # 面
                face_vertices = []
                for part in parts[1:]:
                    # 处理顶点索引（可能包含纹理/法线索引）
                    vertex_index = part.split('/')[0]
                    if vertex_index:
                        try:
                            # OBJ索引从1开始，PLY从0开始
                            face_vertices.append(str(int(vertex_index) - 1))
                        except ValueError:
                            continue

                if len(face_vertices) >= 3:
                    faces.append(face_vertices)
                    face_count += 1

    # 写入PLY文件
    with open(ply_file, 'w') as f:
        # PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertex_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {face_count}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")

        # 写入顶点
        for vertex in vertices:
            f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

        # 写入面
        for face in faces:
            f.write(f"{len(face)} {' '.join(face)}\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: python obj2ply.py <input.obj> <output.ply>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not input_file.lower().endswith('.obj'):
        print("Error: Input file must be an OBJ file (.obj)")
        sys.exit(1)

    obj_to_ply(input_file, output_file)
    print(f"Converted {input_file} to {output_file}")


if __name__ == "__main__":
    main()