#include "OpenGL_TES.hpp"

OpenGL_TES::OpenGL_TES() {}

OpenGL_TES::~OpenGL_TES() {}

void OpenGL_TES::InitOpenGL()
{
    std::cout << "Initializing OpenGL..." << std::endl;
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(window_width, window_height, "CUDA-OpenGL Interop", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        exit(EXIT_FAILURE);
    }

    // glEnable(GL_DEPTH_TEST);

    // Set the beige background color
    // glClearColor(0.96f, 0.96f, 0.86f, 1.0f); // Beige background

    // glMatrixMode(GL_PROJECTION);
    // glLoadIdentity();
    // gluPerspective(45.0, (double)window_width / (double)window_height, 0.1, 100.0);

    // glViewport(0, 0, window_width, window_height);

    // gluLookAt(0.0, 5.0, 5.0,  // Camera position
    //           0.0, 0.0, 0.0,  // Look-at point
    //           0.0, 1.0, 0.0); // Up vector

    // glMatrixMode(GL_MODELVIEW);
    // glLoadIdentity();
}

void OpenGL_TES::RenderGL()
{
    // CreateBuffers();
    std::cout << "OpenGL buffers created successfully." << std::endl;

    while (!glfwWindowShouldClose(window))
    {
        // UpdateBuffers();
        RenderObject();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    std::cout << "Cleaning up OpenGL..." << std::endl;
    glfwDestroyWindow(window);
    glfwTerminate();
}

void OpenGL_TES::Cleanup() {}

/**
 * @brief Creates OpenGL buffers and registers them with CUDA.
 * Called once at the beginning of the rendering.
 */
void OpenGL_TES::CreateBuffers()
{
    std::cout << "CreateBuffers" << std::endl;
    // Create VBO
    // glGenBuffers(1, &vbo);
    // glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW); // 6 floats per vertex (x, y, z, u, v, w)
    // glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    // cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // // Create Texture
    // glGenTextures(1, &texture);
    // glBindTexture(GL_TEXTURE_2D, texture);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 512, 512, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glBindTexture(GL_TEXTURE_2D, 0);

    // // Register Texture with CUDA
    // cudaGraphicsGLRegisterImage(&cuda_texture_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

/**
 * @brief Copies data from CUDA to OpenGL buffers.
 */
void OpenGL_TES::UpdateBuffers()
{
    return;
    // Map VBO
    float4 *d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_vertices, &num_bytes, cuda_vbo_resource);

    // Launch kernel to generate geometry
    // int threads = 256;
    // int blocks = (width * height + threads - 1) / threads;
    // GenerateGeometry<<<blocks, threads>>>(d_vertices, width, height);

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    // // Map Texture
    // cudaArray_t d_texture_array;
    // cudaGraphicsMapResources(1, &cuda_texture_resource, 0);
    // cudaGraphicsSubResourceGetMappedArray(&d_texture_array, cuda_texture_resource, 0, 0);

    // // Launch kernel to generate texture
    // cudaSurfaceObject_t surface;
    // cudaResourceDesc resDesc = {};
    // resDesc.resType = cudaResourceTypeArray;
    // resDesc.res.array.array = d_texture_array;
    // cudaCreateSurfaceObject(&surface, &resDesc);

    // GenerateTexture<<<blocks, threads>>>((uchar4 *)surface, width, height);

    // cudaDestroySurfaceObject(surface);
    // cudaGraphicsUnmapResources(1, &cuda_texture_resource, 0);
}

/**
 * @brief Renders the object using OpenGL.
 */
void OpenGL_TES::RenderObject()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_TRIANGLES);
    glColor3f(1.0f, 0.0f, 0.0f); // Red
    glVertex3f(-0.5f, -0.5f, 0.0f);
    glColor3f(0.0f, 1.0f, 0.0f); // Green
    glVertex3f(0.5f, -0.5f, 0.0f);
    glColor3f(0.0f, 0.0f, 1.0f); // Blue
    glVertex3f(0.0f, 0.5f, 0.0f);
    glEnd();

    /*
    glColor3f(1.0f, 1.0f, 1.0f); // White color

    // Bind the VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);

    // Draw the geometry
    int numVertices = 6;
    glDrawArrays(GL_TRIANGLES, 0, numVertices);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    */
}

int OpenGL_TES::MakeObjectOnGL(std::vector<Facet *> facets)
{
    std::cout << "MakeObjectOnGL: " << facets.size() << std::endl;
    return 1;

    int numFacets = facets.size();

    float4 *vboData = new float4[numFacets * 3];
    for (int i = 0; i < numFacets; ++i)
    {
        Facet *facet = facets[i];
        vboData[i * 3 + 0].x = facet->v1.x;
        vboData[i * 3 + 0].y = facet->v1.y;
        vboData[i * 3 + 0].z = facet->v1.z;
        vboData[i * 3 + 0].w = 1.0f;

        vboData[i * 3 + 1].x = facet->v2.x;
        vboData[i * 3 + 1].y = facet->v2.y;
        vboData[i * 3 + 1].z = facet->v2.z;
        vboData[i * 3 + 1].w = 1.0f;

        vboData[i * 3 + 2].x = facet->v3.x;
        vboData[i * 3 + 2].y = facet->v3.y;
        vboData[i * 3 + 2].z = facet->v3.z;
        vboData[i * 3 + 2].w = 1.0f;

        std::cout << "Facet " << i << ": "
                  << "v1(" << facet->v1.x << ", " << facet->v1.y << ", " << facet->v1.z << "), "
                  << "v2(" << facet->v2.x << ", " << facet->v2.y << ", " << facet->v2.z << "), "
                  << "v3(" << facet->v3.x << ", " << facet->v3.y << ", " << facet->v3.z << ")"
                  << std::endl;
    }

    std::cout << "Copying data to OpenGL..." << std::endl;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * numFacets * 3, vboData, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete[] vboData;
    std::cout << "VBO created successfully." << std::endl;
    return 1;
}