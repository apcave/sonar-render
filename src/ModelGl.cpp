#include "ModelGl.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "OptiX/stb_image_write.h"

void ModelGl::MakeTextureShader()
{
    if (textureShaderProgram != 0)
    {
        return;
    }
    const char *vertexShaderSource = R"(
            #version 330 core

            layout(location = 0) in vec3 aPos;    // Vertex position
            layout(location = 1) in vec2 aTexCoord; // Texture coordinates

            out vec2 TexCoords; // Pass texture coordinates to the fragment shader

            uniform mat4 modelView;      // Model matrix
            uniform mat4 projection;     // Projection matrix

            void main()
            {
                TexCoords = aTexCoord; // Pass texture coordinates to the fragment shader
                //gl_Position = vec4(aPos, 1.0); // Transform vertex position to clip space
                //gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
                gl_Position = projection * modelView * vec4(aPos, 1.0); // Transform vertex position to clip space
            }
    )";

    // Fragment shader
    const char *fragmentShaderSource = R"(
            #version 330 core
            out vec4 FragColor;

            in vec2 TexCoords; // Texture coordinates passed from the vertex shader

            uniform sampler2D inputTexture;

            void main()
            {
                FragColor = texture(inputTexture, TexCoords);
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

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Check for compilation errors
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex Shader Compilation Failed:\n"
                  << infoLog << std::endl;
    }

    // Link shaders into a program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
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

    // Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader Program Linking Failed:\n"
                  << infoLog << std::endl;
    }

    // Clean up shaders (no longer needed after linking)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    textureShaderProgram = shaderProgram;
}

ModelGl::ModelGl()
{
}

ModelGl::~ModelGl() {}

void ModelGl::InitOpenGL()
{
    std::cout << "Initializing OpenGL..." << std::endl;
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (renderImage)
    {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Make the window invisible
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

    glm::mat4 view = glm::lookAt(glm::vec3(5.0f, 5.0f, 5.0f),  // Camera position
                                 glm::vec3(0.0f, 0.0f, 0.0f),  // Look-at point
                                 glm::vec3(0.0f, 1.0f, 0.0f)); // Up vector
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(glm::value_ptr(view));
}

void ModelGl::Cleanup()
{
}

void ModelGl::ProcessFrame()
{
    std::cout << "Processing frame... <------------------------" << std::endl;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    if (!glfwGetCurrentContext())
    {
        std::cerr << "No active OpenGL context!" << std::endl;
        return;
    }

    MakeTextureShader();

    // glBindVertexArray(vao);
    GLfloat projectionMatrix[16];
    GLfloat modelViewMatrix[16];

    // Get the current projection matrix
    glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix);

    // Get the current model-view matrix
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix);

    // Get uniform locations in the shader
    GLint projectionLoc = glGetUniformLocation(textureShaderProgram, "projection");
    GLint modelViewLoc = glGetUniformLocation(textureShaderProgram, "modelView");

    if (projectionLoc == -1)
    {
        std::cerr << "Error: 'projectionLoc' not found in the shader program." << std::endl;
    }

    if (modelViewLoc == -1)
    {
        std::cerr << "Error: 'modelViewLoc' not found in the shader program." << std::endl;
    }

    GLint textureUniformLoc = glGetUniformLocation(textureShaderProgram, "inputTexture");
    if (textureUniformLoc == -1)
    {
        std::cerr << "Error: Uniform 'inputTexture' not found in the shader program." << std::endl;
    }

    if (!renderImage)
    {
        // Pass the matrices to the shader
        glUseProgram(textureShaderProgram);
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projectionMatrix);
        glUniformMatrix4fv(modelViewLoc, 1, GL_FALSE, modelViewMatrix);

        while (!glfwWindowShouldClose(window))
        {
            for (auto object : targetObjects)
            {
                object->RenderObject(textureUniformLoc);
            }

            for (auto object : fieldObjects)
            {
                object->RenderObject(textureUniformLoc);
            }

            // Swap buffers and poll events
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        glBindVertexArray(0);
        std::cout << "Cleaning up OpenGL..." << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return;
    }
    else
    {
        std::cout << "Processing frame for off-screen rendering..." << std::endl;

        // Create a framebuffer
        GLuint framebuffer;
        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        // Create a texture to store the rendered image
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Attach the texture to the framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

        // Check if the framebuffer is complete
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            std::cerr << "Framebuffer is not complete!" << std::endl;
            return;
        }

        glViewport(0, 0, window_width, window_height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Pass the matrices to the shader
        glUseProgram(textureShaderProgram);
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projectionMatrix);
        glUniformMatrix4fv(modelViewLoc, 1, GL_FALSE, modelViewMatrix);

        std::cout << "Rendering to image..." << std::endl;
        // Render to an image or texture instead of a window
        for (auto object : targetObjects)
        {
            object->RenderObject(textureUniformLoc);
        }

        for (auto object : fieldObjects)
        {
            object->RenderObject(textureUniformLoc);
        }

        // Read pixels from the framebuffer
        std::vector<unsigned char> pixels(window_width * window_height * 3); // RGB format
        glReadPixels(0, 0, window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        // Save the image using stb_image_write or another library
        stbi_flip_vertically_on_write(1); // Flip the image vertically
        if (!stbi_write_png("output.png", window_width, window_height, 3, pixels.data(), window_width * 3))
        {
            std::cerr << "Failed to save image!" << std::endl;
        }
        else
        {
            std::cout << "Image saved as 'output.png'" << std::endl;
        }

        // Cleanup
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &framebuffer);
        glDeleteTextures(1, &texture);

        return;
    }
}

int ModelGl::MakeObjectsOnGl()
{
    std::cout << "TODO: Add Objects without Textures." << std::endl;
    for (auto object : targetObjects)
    {
        object->AllocateGl();
    }

    for (auto object : fieldObjects)
    {
        std::cout << "Allocating OpenGL for field object." << std::endl;
        object->AllocateGl();
    }

    return 0;
}
