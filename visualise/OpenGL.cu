#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

void InitOpenGL(GLFWwindow **window)
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // Request OpenGL 4.x
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5); // Request OpenGL 4.5

    *window = glfwCreateWindow(800, 600, "CUDA-OpenGL Interop", NULL, NULL);
    if (!*window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(*window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        exit(EXIT_FAILURE);
    }

    glEnable(GL_DEPTH_TEST);
}

GLuint vbo, texture;
cudaGraphicsResource *cuda_vbo_resource, *cuda_texture_resource;

void CreateBuffers()
{
    // Create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW); // 6 floats per vertex (x, y, z, u, v, w)
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // Create Texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 512, 512, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register Texture with CUDA
    cudaGraphicsGLRegisterImage(&cuda_texture_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

__global__ void GenerateGeometry(float4 *vertices, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height)
        return;

    int x = idx % width;
    int y = idx / width;

    float u = x / (float)(width - 1);
    float v = y / (float)(height - 1);

    vertices[idx] = make_float4(u * 2.0f - 1.0f, v * 2.0f - 1.0f, 0.0f, 1.0f); // x, y, z, w
}

__global__ void GenerateTexture(uchar4 *texture, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height)
        return;

    int x = idx % width;
    int y = idx / width;

    texture[idx] = make_uchar4(x % 256, y % 256, 128, 255); // R, G, B, A
}

void UpdateBuffers(int width, int height)
{
    // Map VBO
    float4 *d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_vertices, &num_bytes, cuda_vbo_resource);

    // Launch kernel to generate geometry
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    GenerateGeometry<<<blocks, threads>>>(d_vertices, width, height);

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    // Map Texture
    cudaArray_t d_texture_array;
    cudaGraphicsMapResources(1, &cuda_texture_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&d_texture_array, cuda_texture_resource, 0, 0);

    // Launch kernel to generate texture
    cudaSurfaceObject_t surface;
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_texture_array;
    cudaCreateSurfaceObject(&surface, &resDesc);

    GenerateTexture<<<blocks, threads>>>((uchar4 *)surface, width, height);

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cuda_texture_resource, 0);
}

void RenderPlate()
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

int test()
{
    // Initialize GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a GLFW window with OpenGL context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);                 // Request OpenGL 4.x
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);                 // Request OpenGL 4.5
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Use core profile
    GLFWwindow *window = glfwCreateWindow(800, 600, "GLEW Test", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the OpenGL context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE; // Enable modern OpenGL features
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Query OpenGL and GPU information
    const GLubyte *renderer = glGetString(GL_RENDERER); // GPU name
    const GLubyte *version = glGetString(GL_VERSION);   // OpenGL version
    std::cout << "Renderer: " << renderer << std::endl;
    std::cout << "OpenGL Version: " << version << std::endl;

    // Clean up and exit
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

int main()
{
    GLFWwindow *window;

    // test();

    InitOpenGL(&window);
    CreateBuffers();

    while (!glfwWindowShouldClose(window))
    {
        UpdateBuffers(512, 512);
        RenderPlate();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}