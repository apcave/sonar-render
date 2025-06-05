#include "ModelGl.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "OptiX/stb_image_write.h"

ModelGl::ModelGl() : eglDisplay(EGL_NO_DISPLAY), eglContext(EGL_NO_CONTEXT), eglSurface(EGL_NO_SURFACE) {}
ModelGl::~ModelGl()
{
    std::cout << "Destroying ModelGl..." << std::endl;
    Cleanup();
}

void ModelGl::InitOpenGL()
{
    std::cout << "Initializing EGL/OpenGL..." << std::endl;

    // gladLoadEGL(eglDisplay, eglGetProcAddress);
    // std::cout << "1 help\n";

    // eglDisplay = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);

    // 1. Get default display
    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    std::cout << "help\n";
    if (eglDisplay == EGL_NO_DISPLAY)
    {
        std::cerr << "Failed to get EGL display" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "EGL display initialized successfully." << std::endl;
    // 2. Initialize EGL
    if (!eglInitialize(eglDisplay, nullptr, nullptr))
    {
        std::cerr << "Failed to initialize EGL" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "test 1\n";

    // 3. Choose EGL config
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE};
    EGLConfig eglConfig;
    EGLint numConfigs;
    if (!eglChooseConfig(eglDisplay, configAttribs, &eglConfig, 1, &numConfigs) || numConfigs == 0)
    {
        std::cerr << "Failed to choose EGL config" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Test 2\n";

    // 4. Create a PBuffer surface
    EGLint pbufferAttribs[] = {
        EGL_WIDTH,
        window_width,
        EGL_HEIGHT,
        window_height,
        EGL_NONE,
    };
    eglSurface = eglCreatePbufferSurface(eglDisplay, eglConfig, pbufferAttribs);
    if (eglSurface == EGL_NO_SURFACE)
    {
        std::cerr << "Failed to create EGL PBuffer surface" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Test 3\n";

    // 5. Bind OpenGL API
    if (!eglBindAPI(EGL_OPENGL_API))
    {
        std::cerr << "Failed to bind OpenGL API" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Test 4\n";

    // 6. Create OpenGL context
    EGLint ctxAttribs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE};
    eglContext = eglCreateContext(eglDisplay, eglConfig, EGL_NO_CONTEXT, ctxAttribs);
    if (eglContext == EGL_NO_CONTEXT)
    {
        std::cerr << "Failed to create EGL context" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Test 5\n";

    // 7. Make context current
    if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext))
    {
        std::cerr << "Failed to make EGL context current" << std::endl;
        exit(EXIT_FAILURE);
    }

    // std::cout << "OOPS OpenGL context initialized successfully." << std::endl;
    // if (!gladLoadGL((GLADloadfunc)eglGetProcAddress))
    // {
    //     std::cerr << "Failed to initialize GLAD" << std::endl;
    //     exit(EXIT_FAILURE);
    // }
    std::cout << "EGL OpenGL context initialized successfully." << std::endl;

    // OpenGL state setup
    glClearColor(0.96f, 0.96f, 0.86f, 1.0f); // Beige background

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)window_width / (float)window_height, 0.1f, 100.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(glm::value_ptr(projection));

    float dist = 12.0f;
    glm::mat4 view = glm::lookAt(glm::vec3(dist, dist, dist),  // Camera position
                                 glm::vec3(0.0f, 0.0f, 0.0f),  // Look-at point
                                 glm::vec3(0.0f, 1.0f, 0.0f)); // Up vector
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(glm::value_ptr(view));

    const GLubyte *version = glGetString(GL_VERSION);
    if (!version)
    {
        std::cerr << "No OpenGL context is current! glGenBuffers will segfault." << std::endl;
        abort();
    }
    else
    {
        std::cout << "OpenGL context version: " << version << std::endl;
    }
}

void ModelGl::Cleanup()
{
    // std::cout << "Cleaning up EGL/OpenGL..." << std::endl;
    // if (eglDisplay != EGL_NO_DISPLAY)
    // {
    //     eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    //     if (eglContext != EGL_NO_CONTEXT)
    //         eglDestroyContext(eglDisplay, eglContext);
    //     if (eglSurface != EGL_NO_SURFACE)
    //         eglDestroySurface(eglDisplay, eglSurface);
    //     eglTerminate(eglDisplay);
    // }
    // eglDisplay = EGL_NO_DISPLAY;
    // eglContext = EGL_NO_CONTEXT;
    // eglSurface = EGL_NO_SURFACE;
    // std::cout << "Completed..." << std::endl;
}

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

void ModelGl::ProcessFrame()
{
    std::cout << "Processing frame... <------------------------" << std::endl;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    if (eglGetCurrentContext() == EGL_NO_CONTEXT)
    {
        std::cout << "Error: No current EGL OpenGL context before creating a texture." << std::endl;
        return;
    }

    // // --- Ambient light setup ---
    // GLfloat ambient_light[] = {0.8f, 0.8f, 0.8f, 1.0f}; // RGBA, adjust as needed
    // glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_light);

    // // Enable lighting and at least one light source
    // glEnable(GL_LIGHTING);
    // glEnable(GL_LIGHT0);

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
}

int ModelGl::MakeObjectsOnGl()
{
    std::cout << "TODO: Add Objects without Textures." << std::endl;
    for (auto object : targetObjects)
    {
        std::cout << "Allocating OpenGL for target object." << std::endl;
        object->AllocateGl();
    }

    for (auto object : fieldObjects)
    {
        std::cout << "Allocating OpenGL for field object." << std::endl;
        object->AllocateGl();
    }

    return 0;
}

void ModelGl::FreeGl()
{
    std::cout << "TODO: Freeing Objects without Textures." << std::endl;
    for (auto object : targetObjects)
    {
        object->FreeGl();
    }

    for (auto object : fieldObjects)
    {
        object->FreeGl();
    }

    std::cout << "Cleaning up EGL/OpenGL..." << std::endl;
    if (eglDisplay != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (eglContext != EGL_NO_CONTEXT)
            eglDestroyContext(eglDisplay, eglContext);
        if (eglSurface != EGL_NO_SURFACE)
            eglDestroySurface(eglDisplay, eglSurface);
        eglTerminate(eglDisplay);
    }
    eglDisplay = EGL_NO_DISPLAY;
    eglContext = EGL_NO_CONTEXT;
    eglSurface = EGL_NO_SURFACE;
    std::cout << "Completed..." << std::endl;
}
