
def write_vertices_to_node(obj_file, node_file):
    """Convert OBJ vertices to node format"""
    vertices = []

    with open(obj_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])

    with open(node_file, 'w') as f:
        # Header
        f.write(f"# Node count, 3 dim, no attribute, no boundary marker\n")
        f.write(f"{len(vertices)} 3 0 0\n")
        f.write("# Node index, node coordinates\n")

        # Vertices
        for i, v in enumerate(vertices, 1):
            f.write(f"{i} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python obj2node.py <input.obj> <output.node>")
        sys.exit(1)

    input_obj = sys.argv[1]
    output_node = sys.argv[2]

    write_vertices_to_node(input_obj, output_node)
    print(f"Converted {input_obj} to {output_node}")