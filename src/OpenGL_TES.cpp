#include "OpenGL_TES.hpp"

OpenGL_TES::OpenGL_TES() {}

OpenGL_TES::~OpenGL_TES() {}

void OpenGL_TES::InitOpenGL()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // Request OpenGL 4.x
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5); // Request OpenGL 4.5

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

    glEnable(GL_DEPTH_TEST);
}

void OpenGL_TES::RenderGL()
{

    InitOpenGL();
    std::cout << "OpenGL initialized successfully." << std::endl;

    CreateBuffers();
    std::cout << "OpenGL buffers created successfully." << std::endl;

    while (!glfwWindowShouldClose(window))
    {
        UpdateBuffers();
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
    // Create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW); // 6 floats per vertex (x, y, z, u, v, w)
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

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

    // Bind VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);

    // Bind Texture
    glBindTexture(GL_TEXTURE_2D, texture);

    // Draw the plate
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}
