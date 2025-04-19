#include "OpenGL_TES.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void OpenGL_TES::MakeTextureShader()
{
    if (textureShaderProgram != 0)
    {
        return;
    }

    // Fragment shader
    const char *fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;

        in vec2 TexCoords;

        uniform sampler2D inputTexture;
        uniform float textureScaler;

        void main()
        {
            vec4 texColor = texture(inputTexture, TexCoords);
            FragColor = texColor * textureScaler;
        }
    )";
    GLint success;
    char infoLog[512];

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Check for compilation errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment Shader Compilation Failed:\n"
                  << infoLog << std::endl;
    }

    // Link shaders into a program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, fragmentShader);
    // glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {

        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "textureShaderProgram Program Linking Failed:\n"
                  << infoLog << std::endl;
    }
    std::cout << "textureShaderProgram program linked successfully." << std::endl;

    // Clean up shaders (no longer needed after linking)
    glDeleteShader(fragmentShader);

    textureShaderProgram = shaderProgram;
}

void OpenGL_TES::MakeUniformShader()
{
    if (uniformShaderProgram != 0)
    {
        return;
    }
    const char *fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;

        uniform vec4 uniformColor;

        void main()
        {
            FragColor = uniformColor;
        }
    )";
    GLint success;
    char infoLog[512];

    // Compile the vertex shader
    GLuint fragmentShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Check for vertex shader compilation errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Vertex Shader Compilation Failed:\n"
                  << infoLog << std::endl;
    }

    // Link shaders into a program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {

        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader Program Linking Failed:\n"
                  << infoLog << std::endl;
    }
    std::cout << "uniformShaderProgram linked successfully." << std::endl;

    // Clean up shaders (no longer needed after linking)
    glDeleteShader(fragmentShader);
    uniformShaderProgram = shaderProgram;
}

OpenGL_TES::OpenGL_TES()
{
}

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

    glClearColor(0.96f, 0.96f, 0.86f, 1.0f); // Beige background

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)window_width / (float)window_height, 0.1f, 100.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(glm::value_ptr(projection));

    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f),  // Camera position
                                 glm::vec3(0.0f, 0.0f, 0.0f),  // Look-at point
                                 glm::vec3(0.0f, 1.0f, 0.0f)); // Up vector
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(glm::value_ptr(view));
}

void OpenGL_TES::RenderGL()
{
    // CreateBuffers();
    // std::cout << "OpenGL buffers created successfully." << std::endl;

    // while (!glfwWindowShouldClose(window))
    // {
    //     // UpdateBuffers();
    //     RenderObject();

    //     glfwSwapBuffers(window);
    //     glfwPollEvents();
    // }

    // std::cout << "Cleaning up OpenGL..." << std::endl;
    // glfwDestroyWindow(window);
    // glfwTerminate();
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

    // // // Map Texture
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
    std::cout << "RenderObject----------------------------------" << std::endl;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // glColor3f(1.0f, 1.0f, 1.0f); // White color

    // // Bind the VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);

    // Draw the geometry
    int numVertices = 6;
    glDrawArrays(GL_TRIANGLES, 0, numVertices);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int OpenGL_TES::MakeObjectOnGL(std::vector<Facet *> facets)
{

    std::vector<float> vertexData;
    for (auto &facet : facets)
    {
        // Add vertex positions and texture coordinates for each facet
        vertexData.insert(vertexData.end(), {facet->v1.x, facet->v1.y, facet->v1.z, facet->texCoords[0], facet->texCoords[1],
                                             facet->v2.x, facet->v2.y, facet->v2.z, facet->texCoords[2], facet->texCoords[3],
                                             facet->v3.x, facet->v3.y, facet->v3.z, facet->texCoords[4], facet->texCoords[5]});
    }

    // Upload the vertex data to the GPU
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    std::cout << "VBO created successfully." << std::endl;

    return 1;
}

void OpenGL_TES::CreateTexture(int numXpnts, int numYpnts, GLuint *texture)
{
    std::cout << "CreateTexture for Single-Channel Float Texture" << std::endl;
    if (!glfwGetCurrentContext())
    {
        std::cout << "Error Intializing OpenGL context before creating a texture." << std::endl;
        InitOpenGL();
    }

    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);

    // Allocate memory for a single-channel float texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, numXpnts, numYpnts, 0, GL_RED, GL_FLOAT, NULL);

    // Set texture filtering parameters
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Unbind the texture
    glBindTexture(GL_TEXTURE_2D, 0);

    std::cout << "Single-channel float texture created successfully." << std::endl;
}
void OpenGL_TES::ProcessFrame()
{
    std::cout << "Processing frame... <------------------------" << std::endl;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glDisable(GL_DEPTH_TEST);

    if (!glfwGetCurrentContext())
    {
        std::cerr << "No active OpenGL context!" << std::endl;
        return;
    }

    MakeTextureShader();

    // Ensure VAO is initialized
    if (vao == 0)
    {
        std::cout << "Intializing VAO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
        // Vertex Attribute Object....
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);

        // Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Unbind the VBO and VAO
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    std::cout << "Processing frame..." << std::endl;

    glUseProgram(textureShaderProgram);

    // Get the memory addresses for input variables.
    GLint textureUniformLocation = glGetUniformLocation(textureShaderProgram, "inputTexture");
    if (textureUniformLocation == -1)
    {
        std::cerr << "Error: Uniform 'inputTexture' not found in the shader program." << std::endl;
    }
    else
    {
        std::cout << "Uniform 'inputTexture' location: " << textureUniformLocation << std::endl;
    }

    GLint textureScalerLocation = glGetUniformLocation(textureShaderProgram, "textureScaler");
    if (textureScalerLocation == -1)
    {
        std::cerr << "Error: Uniform 'textureScaler' not found in the shader program." << std::endl;
    }
    else
    {
        std::cout << "Uniform 'textureScaler' location: " << textureUniformLocation << std::endl;
    }

    for (auto facGL : gl_object_facets)
    {
        // Unmap the CUDA resource for this facet's texture
        cudaGraphicsUnmapResources(1, &facGL->cudaResource, 0);
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float *mappedData = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    if (mappedData)
    {
        // Access the vertex data
        for (size_t i = 0; i < 6 * 5; i += 5) // Assuming 3 position + 2 texture coordinates
        {
            std::cout << "Position: (" << mappedData[i] << ", " << mappedData[i + 1] << ", " << mappedData[i + 2] << ")";
            std::cout << ", TexCoords: (" << mappedData[i + 3] << ", " << mappedData[i + 4] << ")" << std::endl;
        }

        if (glUnmapBuffer(GL_ARRAY_BUFFER) == GL_FALSE)
        {
            std::cerr << "Warning: Buffer was corrupted and could not be unmapped." << std::endl;
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    else
    {
        std::cerr << "Failed to map buffer." << std::endl;
    }

    glBindVertexArray(vao);

    int triangleIndex = 0;
    while (!glfwWindowShouldClose(window))
    {
        for (auto facGL : gl_object_facets)
        {

            // Bind the texture for this facet
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, facGL->textureID);
            glUniform1i(textureUniformLocation, 0);

            glUniform1f(textureScalerLocation, 0.5f);

            glDrawArrays(GL_TRIANGLES, triangleIndex * 3, 3); // Render 3 vertices for the facet
            triangleIndex++;
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glBindVertexArray(0);
    std::cout << "Cleaning up OpenGL..." << std::endl;
    glfwDestroyWindow(window);
    glfwTerminate();
}