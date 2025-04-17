# Dataset

# RAW DATA

I have original data in form of DAT files corresponding to one timestep per dat files, and a collection of all the timesteps for each mass ratio experiment. First I will extract this raw data and then extract physical features from this data that will be then arranged for the spatiotemporal classifier later.

for each mass ratio, and for each reduced speed u, we have DAT FILES for each timestep containing the values of (x, y, z, u_vel, v_vel, w_vel, P).

Here's how the information is stored in the original dat file:

“I have the following data with me in the form of a dat file:

"TITLE     = "3D Mesh"
VARIABLES = "X"
"Y"
"Z"
"uvel" "vvel" "wvel" "pressure"
ZONE T="FEBRICK"
STRANDID=1, SOLUTIONTIME=   5.00000000

N=      107894 , E=      107016  ZONETYPE=FEBRICK
DATAPACKING=POINT
-0.403995787E-01   0.477070432E+00   0.000000000E+00  -0.152023650E-02   0.117254718E+01   0.000000000E+00  -0.152493046E+00
-0.250000000E+02  -0.150000000E+02   0.000000000E+00   0.000000000E+00   0.100000000E+01   0.000000000E+00   0.235365858E-01"
...

"0.333551519E+00   0.340522290E+00   0.100000000E+01  -0.789242085E-02  -0.899476993E-02   0.000000000E+00   0.208092480E+00
0.329151863E+00   0.342898208E+00   0.100000000E+01  -0.795626102E-02  -0.911258492E-02   0.000000000E+00   0.407104159E+00
3080     3083     3486     3486    57027    57030    57433    57433
2139     2117     2140     2140    56086    56064    56087    56087
2116     2145     2146     2146    56063    56092    56093    56093"

---

The file provided is an ASCII data file formatted in a style very similar to the Tecplot format, which is often used to store and visualize simulation meshes and associated field data. Here’s a breakdown of its structure and contents:

---

### Header Information

- **Title and Variables:**
    
    The file begins with a header block that defines the overall title of the dataset and lists the variable names:
    
    - **TITLE:** “3D Mesh”
    - **VARIABLES:** It lists the spatial coordinates `"X"`, `"Y"`, and `"Z"`, followed by physical variables `"uvel"`, `"vvel"`, `"wvel"`, and `"pressure"`.
        
        These indicate that each data point will include a position in 3D space and associated values such as velocity components and pressure.
        
- **Zone Definition:**
    
    The header also specifies a **ZONE** with additional attributes:
    
    - **T="FEBRICK":** This tag not only names the zone but hints at the element type.
    - **STRANDID=1, SOLUTIONTIME=5.00000000:** These are metadata items; the strand ID may be used to track different parts or time steps in a simulation, and the solution time (here, 5.0 units) indicates the time level corresponding to the data.
    - **N and E:**
        - **N=107894:** This is the number of nodes (or points) in the mesh.
        - **E=107016:** This represents the number of elements (likely hexahedral “bricks”).
    - **ZONETYPE=FEBRICK:** This confirms that the zone is made up of finite elements in brick (hexahedral) form.
    - **DATAPACKING=POINT:** This means that the data is organized so that for each node all variable values are provided together (i.e., each row or record corresponds to one mesh point).

---

### Data Sections

- **Node (Point) Data:**
    
    After the header, you see lines containing numerical values in scientific notation. Each line (or group of values) corresponds to a single node, listing:
    
    - The spatial coordinates (X, Y, Z)
    - The associated field values (uvel, vvel, wvel, pressure)
        
        For example, one line shows:
        
        -0.403995787E-01   0.477070432E+00   0.000000000E+00  -0.152023650E-02   0.117254718E+01   0.000000000E+00  -0.152493046E+00
        
    
    This is one node’s complete set of data.
    
- **Connectivity (Element) Data:**
    
    Later in the file, blocks of integers appear. In a FEBRICK (hexahedral element) zone, connectivity information defines which nodes form each element. Although the snippet only shows a few lines, these integers represent the node indices that, when grouped appropriately (typically 8 per element for a brick), define the topology of each finite element. The connectivity data might be organized in columns or in a structured block that maps to the mesh’s element connectivity.
    

# Metadata Generation

## Reconstructing Gradients for Flow Feature Computation

To calculate vorticity, strain rate, and pressure gradient from your data, we need to use WLS to reconstruct all velocity component gradients. Here's the mathematical approach:

### Weighted Least Squares (WLS) Gradient Reconstruction

For any scalar field φ (u, v, w, or pressure) at a node i:

1. For each node i with position vector xi:
    
    xi\mathbf{x}_i
    
    - Identify all neighboring nodes j with positions xj
        
        xj\mathbf{x}_j
        
    - Define relative position vectors Δxij=xj−xi
        
        Δxij=xj−xi\Delta \mathbf{x}_{ij} = \mathbf{x}_j - \mathbf{x}_i
        
    - Calculate value differences Δϕij=ϕj−ϕi
        
        Δϕij=ϕj−ϕi\Delta \phi_{ij} = \phi_j - \phi_i
        
2. Set up the overdetermined system:
    - Each neighbor j contributes one equation: ∇ϕi⋅Δxij≈Δϕij
        
        ∇ϕi⋅Δxij≈Δϕij\nabla \phi_i \cdot \Delta \mathbf{x}_{ij} \approx \Delta \phi_{ij}
        
    - Apply weights wij=∣Δxij∣21 to prioritize closer neighbors
        
        wij=1∣Δxij∣2w_{ij} = \frac{1}{|\Delta \mathbf{x}_{ij}|^2}
        
3. Formulate the weighted least squares problem:
    - Minimize ∑jwij(∇ϕi⋅Δxij−Δϕij)2
        
        ∑jwij(∇ϕi⋅Δxij−Δϕij)2\sum_j w_{ij}(\nabla \phi_i \cdot \Delta \mathbf{x}_{ij} - \Delta \phi_{ij})^2
        
    - Solution is ∇ϕi=(ATWA)−1ATWb where:
        
        ∇ϕi=(ATWA)−1ATWb\nabla \phi_i = (A^T W A)^{-1} A^T W \mathbf{b}
        
        - AA
        A is the matrix of position differences
        - WW
        W is the diagonal weight matrix
        - b\mathbf{b}
        b is the vector of value differences

### Feature Computation

Once we have the gradients, we compute:

### Vorticity Vector Components

ωx=∂w∂y−∂v∂z\omega_x = \frac{\partial w}{\partial y} - \frac{\partial v}{\partial z}
ωx=∂y∂w−∂z∂v

ωy=∂u∂z−∂w∂x\omega_y = \frac{\partial u}{\partial z} - \frac{\partial w}{\partial x}
ωy=∂z∂u−∂x∂w

ωz=∂v∂x−∂u∂y\omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
ωz=∂x∂v−∂y∂u

### Strain Rate Tensor

Sij=12(∂ui∂xj+∂uj∂xi)S_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)
Sij=21(∂xj∂ui+∂xi∂uj)

Where uiu_i
ui is the i-th velocity component and xjx_j
xj is the j-th coordinate.

For 3D flow:

- Sxx=∂u∂xS_{xx} = \frac{\partial u}{\partial x}
Sxx=∂x∂u
- Syy=∂v∂yS_{yy} = \frac{\partial v}{\partial y}
Syy=∂y∂v
- Szz=∂w∂zS_{zz} = \frac{\partial w}{\partial z}
Szz=∂z∂w
- Sxy=Syx=12(∂u∂y+∂v∂x)S_{xy} = S_{yx} = \frac{1}{2}\left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\right)
Sxy=Syx=21(∂y∂u+∂x∂v)
- Sxz=Szx=12(∂u∂z+∂w∂x)S_{xz} = S_{zx} = \frac{1}{2}\left(\frac{\partial u}{\partial z} + \frac{\partial w}{\partial x}\right)
Sxz=Szx=21(∂z∂u+∂x∂w)
- Syz=Szy=12(∂v∂z+∂w∂y)S_{yz} = S_{zy} = \frac{1}{2}\left(\frac{\partial v}{\partial z} + \frac{\partial w}{\partial y}\right)
Syz=Szy=21(∂z∂v+∂y∂w)

Scalar strain rate magnitude:
∣S∣=2SijSij=2(Sxx2+Syy2+Szz2+2Sxy2+2Sxz2+2Syz2)|S| = \sqrt{2S_{ij}S_{ij}} = \sqrt{2(S_{xx}^2 + S_{yy}^2 + S_{zz}^2 + 2S_{xy}^2 + 2S_{xz}^2 + 2S_{yz}^2)}
∣S∣=2SijSij=2(Sxx2+Syy2+Szz2+2Sxy2+2Sxz2+2Syz2)

### Pressure Gradient Vector

∇P=(∂P∂x,∂P∂y,∂P∂z)\nabla P = \left(\frac{\partial P}{\partial x}, \frac{\partial P}{\partial y}, \frac{\partial P}{\partial z}\right)
∇P=(∂x∂P,∂y∂P,∂z∂P)

Pressure gradient magnitude:
∣∇P∣=(∂P∂x)2+(∂P∂y)2+(∂P∂z)2|\nabla P| = \sqrt{\left(\frac{\partial P}{\partial x}\right)^2 + \left(\frac{\partial P}{\partial y}\right)^2 + \left(\frac{\partial P}{\partial z}\right)^2}
∣∇P∣=(∂x∂P)2+(∂y∂P)2+(∂z∂P)2

## Metadata Generation Code Review:

This code represents a CFD (Computational Fluid Dynamics) post-processing tool designed to extract advanced flow features from existing simulation data. The program reads Tecplot-format data files containing velocity and pressure fields, calculates derivative quantities (vorticity, strain rate, pressure gradients), and outputs these as new data files with the same format structure.

## Code Structure Overview

The code is organized into several key functions that form a data processing pipeline:

1. **File Parsing**: `parse_mesh_file()` reads and extracts data from Tecplot-format files
2. **Gradient Computation**: `build_neighbors()` and `compute_weighted_gradient()` implement a weighted least-squares gradient reconstruction
3. **Feature Extraction**: `compute_metadata()` calculates vorticity, strain rate, and pressure gradients
4. **File Writing**: `write_new_dat_file()` outputs the computed data
5. **Process Management**: `process_file()` and `main()` handle file selection and overall workflow

## Detailed Function Analysis

### `parse_mesh_file(filename)`

This function reads a Tecplot-format CFD data file and extracts the mesh geometry, flow field data, and element connectivity.

**Key operations:**

- Separates header and data sections of the file
- Extracts node count (N) and element count (E) using regex
- Parses coordinate data (X,Y,Z) and flow variables (u,v,w,p) for each node
- Processes element connectivity information (which nodes form each element)
- Returns properly formatted arrays for further processing

The function handles Tecplot's typical format where header metadata precedes a section of nodal data (coordinates and field values), followed by element connectivity information.

### `build_neighbors(connectivity, num_nodes)`

This utility function constructs a neighbor map for each node by analyzing the element connectivity data.

**Key operations:**

- Creates a dictionary mapping each node index to a set of its neighboring nodes
- Loops through each element and identifies connected nodes
- This neighborhood information is critical for the gradient reconstruction algorithm

### `compute_weighted_gradient(coords, field, neighbors)`

This function implements a weighted least-squares approach to reconstruct spatial gradients of scalar fields.

**Key operations:**

- For each node, constructs a system of equations using neighboring nodes
- Each neighbor contributes an equation relating the field gradient to the observed field difference
- Weight factor 1/|Δx|² gives closer neighbors more influence
- Solves this overdetermined system using least squares (numpy.linalg.lstsq)
- Returns an array of gradient vectors (∂φ/∂x, ∂φ/∂y, ∂φ/∂z) for each node

The gradient reconstruction is fundamental to computing vorticity and strain rate from velocity components.

### `compute_metadata(coords, uvel, vvel, wvel, pressure, connectivity)`

This function calculates advanced flow features based on the velocity and pressure fields.

**Key operations:**

- Reconstructs gradients for each velocity component (u,v,w) and pressure
- Computes vorticity vector (ω = ∇ × u) components:
    - ωₓ = ∂w/∂y - ∂v/∂z
    - ωᵧ = ∂u/∂z - ∂w/∂x
    - ωᵤ = ∂v/∂x - ∂u/∂y
- Calculates strain rate tensor components:
    - Diagonal: Sxx, Syy, Szz (normal strain rates)
    - Off-diagonal: Sxy, Sxz, Syz (shear strain rates)
- Computes strain rate magnitude: √(2(Sxx² + Syy² + Szz² + 2Sxy² + 2Sxz² + 2Syz²))
- Extracts pressure gradient vector components
- Returns a combined data array with all computed quantities

This function essentially transforms the raw velocity and pressure data into more insightful flow characteristics.

### `write_new_dat_file(filename, header_lines, new_node_data, connectivity)`

This function writes the computed metadata to a new Tecplot-format file.

**Key operations:**

- Updates the file header to reflect new variable names
- Writes new node data in scientific notation for precision
- Converts connectivity information back to 1-indexing format (Tecplot standard)
- Maintains file structure compatible with Tecplot visualization software

### `process_file(input_filename, output_filename)`

This function orchestrates the processing of a single input file.

**Key operations:**

- Parses the input file
- Computes flow metadata
- Writes results to the output file

### `main()`

The main function handles file selection and batch processing.

**Key operations:**

- Uses glob pattern matching to find input files in a specified folder
- Uses regex to extract timestep information from filenames
- Generates appropriate output filenames
- Processes each file and handles errors gracefully
- Provides progress feedback

## Technical Details

### Gradient Reconstruction Method

The code uses a weighted least-squares approach for gradient reconstruction, which is well-suited for unstructured meshes where nodes may not be arranged in a regular grid:

1. For each node i, all neighboring nodes j contribute equations of the form:
    - ∇φᵢ · (xⱼ - xᵢ) ≈ (φⱼ - φᵢ)
    - Where ∇φᵢ is the gradient to be determined
    - (xⱼ - xᵢ) is the displacement vector
    - (φⱼ - φᵢ) is the observed field difference
2. Each equation is weighted by wᵢⱼ = 1/|xⱼ - xᵢ|², giving more influence to closer neighbors
3. The system is solved using least squares, which is appropriate since there are typically more neighbors (equations) than dimensions (3 unknowns)

This approach provides robust gradient estimates even on irregular meshes.

### Flow Features Calculated

1. **Vorticity (ω = ∇ × u)**: A vector field measuring local rotation in the fluid. High vorticity indicates swirling flow structures.
2. **Strain Rate**: Measures the rate of deformation of fluid elements. The strain rate tensor captures how quickly fluid elements are stretching, compressing, and shearing.
3. **Pressure Gradient**: Indicates the direction and magnitude of pressure changes. Important for understanding flow acceleration and forces.

## Implementation Notes

1. **Error Handling**: The code includes checks for data consistency and appropriate error messages.
2. **File Pattern Matching**: Uses regex and glob for flexible file selection.
3. **Memory Efficiency**: Processes one file at a time rather than loading all files simultaneously.
4. **Numerical Precision**: Uses scientific notation with 9 decimal places for output.
5. **Indexing Convention**: Converts between 0-indexing (Python) and 1-indexing (Tecplot format).

# DATA PREPERATION

Final objective of the python is to organize data for a particular mass ratio object in the following format also defined in the approach above: 

1. **Per Timestep (Raw Data)**:
    - **Nodes**: Each node has:
        - Coordinates: (x, y, z) (used to define edges or optionally as features).
        - Features: Physical properties [ω_x, ω_y, ω_z, pressure_grad_x, pressure_grad_y, pressure_grad_z, strain_rate_1, ...], totaling F features (e.g., if you have 3 vorticity components, 3 pressure gradients, and 2 strain rates, F = 8).
        - Shape: (N_t, F) where N_t is the number of nodes at timestep t (may vary).
    - **Edges**: Connectivity between nodes, derived from your mesh (e.g., existing mesh edges or computed via Delaunay triangulation).
        - Shape: (2, num_edges_t) (e.g., [[node_i, node_j], ...]).
2. **Per Mass Ratio (Single Data Object)**:
    - **Structure**: A collection of T graphs, one per timestep, where T is the number of timesteps for that mass ratio.
    - **Components**:
        - **Node Features**: List of T tensors, each (N_t, F).
        - **Edge Indices**: List of T edge arrays, each (2, num_edges_t).
    - **Purpose**: This object encapsulates all spatial (mesh) and temporal (timesteps) data for a mass ratio.

### Final Data Format

- **Per Mass Ratio Object**: A dictionary with:
    - 'nodes': List of T tensors, each (N_t, F).
    - 'edges': List of T tensors, each (2, num_edges_t).
- **Stored**: In HDF5 as described.

The HDF5 file will contain a group called `"mass_ratio"` with subgroups for each timestep (named like `"timestep_0001"`, `"timestep_0025"`, etc.). Inside each subgroup are two datasets:

- `"nodes"`: a (Nₜ, F) array of the node features (here, F = 7).
- `"edges"`: a (2, num_edgesₜ) array of the connectivity edges.

## Data Preparation Code Review:

This code is designed to process computational fluid dynamics (CFD) data, specifically related to beam simulations with varying mass ratios. The code reads Tecplot-format data files, extracts node data and connectivity information, and organizes this information into a structured format saved as an HDF5 file. Let me analyze each component in detail.

## Overall Structure and Purpose

The code processes simulation data for fluid-structure interaction problems, focusing on:

1. Reading Tecplot-format metadata files
2. Extracting physical features (vorticity, strain, pressure gradients)
3. Computing mesh connectivity (edges between nodes)
4. Organizing data temporally across multiple timesteps
5. Saving the processed data in a structured HDF5 file

This preprocessing step likely prepares data for subsequent analysis, visualization, or machine learning applications.

## Key Components

### 1. `parse_metadata_file(filename)`

This function parses a Tecplot-format data file (.dat) and extracts three key pieces of information:

- **Header lines**: Metadata about the simulation (preserved as strings)
- **Node data**: A NumPy array with shape (N, 10) containing:
    - Spatial coordinates (X, Y, Z) - columns 0-2
    - Vorticity components (ω_x, ω_y, ω_z) - columns 3-5
    - Strain magnitude - column 6
    - Pressure gradient components (∇p_x, ∇p_y, ∇p_z) - columns 7-9
- **Connectivity array**: A NumPy array with shape (E, 8) representing element connectivity

The function:

1. Separates header and data sections
2. Extracts the number of nodes (N) and elements (E) using regex
3. Parses the data into tokens
4. Shapes the node data into a structured array
5. Converts the connectivity data to 0-indexed format (from Tecplot's 1-indexed format)

### 2. `extract_features(node_data)`

A simple utility function that separates physical properties from spatial coordinates:

- Takes the full node_data array (N×10)
- Returns only the physical features (columns 3-9, shape N×7)

### 3. `compute_edges(connectivity)`

This function:

1. Converts the element connectivity information to an edge list
2. For each hexahedral element (8 nodes), identifies all possible pairs of nodes as edges
3. Uses a set to eliminate duplicate edges
4. Returns a (2×num_edges) array where each column represents one edge

### 4. `process_metadata_file(filename)`

A wrapper function that:

1. Parses a single metadata file
2. Extracts the features
3. Computes the edges
4. Returns the processed data

### 5. `create_mass_ratio_object(folder)`

This function processes all metadata files in a specified folder:

1. Finds all files matching the pattern "beam2_metadata_*.dat"
2. Sorts them by timestep number
3. Processes each file to extract features and edges
4. Organizes the data into a dictionary with:
    - `timesteps`: List of timestep numbers
    - `nodes`: List of feature arrays (one per timestep)
    - `edges`: List of edge arrays (one per timestep)

### 6. `save_mass_ratio_object(mass_ratio_object, output_filename)`

This function saves the processed data to an HDF5 file with a structured layout:

- Creates a "mass_ratio" group
- For each timestep, creates a subgroup "timestep_XXXX"
- In each timestep group, stores:
    - "nodes": Node feature array
    - "edges": Edge connectivity array

### 7. `main()`

The main function:

1. Parses command-line arguments for input folder and output file
2. Creates the mass ratio object by processing all data files
3. Saves the result to the specified HDF5 file

## Technical Aspects

### Data Format

The code assumes:

- **Input**: Tecplot-format .dat files with a specific structure
- Each file represents one timestep of a beam simulation
- Each node has 10 data values (3 spatial + 7 physical properties)
- Elements are FEBRICK type (hexahedral with 8 nodes)

### File Naming Convention

The code expects files named `beam2_metadata_XXXX.dat` where `XXXX` is the timestep number.

### Data Organization

The output HDF5 file organizes data hierarchically:

```
mass_ratio/
    timestep_0001/
        nodes    # Array (N₁, 7)
        edges    # Array (2, E₁)
    timestep_0002/
        nodes    # Array (N₂, 7)
        edges    # Array (2, E₂)
    ...

```

### Physical Properties

The code extracts seven physical properties:

1. Vorticity x-component (ω_x)
2. Vorticity y-component (ω_y)
3. Vorticity z-component (ω_z)
4. Pressure gradient x-component (∇p_x)
5. Pressure gradient y-component (∇p_y)
6. Pressure gradient z-component (∇p_z)
7. Strain rate magnitude
