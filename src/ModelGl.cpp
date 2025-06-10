#include "ModelGl.hpp"

#include <glm/gtc/type_ptr.hpp>
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "OptiX/stb_image_write.h"

ModelGl::ModelGl() : eglDisplay(EGL_NO_DISPLAY), eglContext(EGL_NO_CONTEXT), eglSurface(EGL_NO_SURFACE) {}
ModelGl::~ModelGl()
{
}

void ModelGl::InitOpenGL(int width, int height, float viewSettings[9])
{

    if (eglDisplay != EGL_NO_DISPLAY)
    {
        return;
    }

    std::cout << "Initializing EGL/OpenGL..." << std::endl;

    // 1. Get default display
    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDisplay == EGL_NO_DISPLAY)
    {
        std::cerr << "Failed to get EGL display" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 2. Initialize EGL
    if (!eglInitialize(eglDisplay, nullptr, nullptr))
    {
        std::cerr << "Failed to initialize EGL" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 3. Choose EGL config
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24, // <-- Add this line!
        EGL_NONE};
    EGLConfig eglConfig;
    EGLint numConfigs;
    if (!eglChooseConfig(eglDisplay, configAttribs, &eglConfig, 1, &numConfigs) || numConfigs == 0)
    {
        std::cerr << "Failed to choose EGL config" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 4. Create a PBuffer surface
    EGLint pbufferAttribs[] = {
        EGL_WIDTH,
        width,
        EGL_HEIGHT,
        height,
        EGL_NONE,
    };
    eglSurface = eglCreatePbufferSurface(eglDisplay, eglConfig, pbufferAttribs);
    if (eglSurface == EGL_NO_SURFACE)
    {
        std::cerr << "Failed to create EGL PBuffer surface" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 5. Bind OpenGL API
    if (!eglBindAPI(EGL_OPENGL_API))
    {
        std::cerr << "Failed to bind OpenGL API" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 6. Create OpenGL context
    EGLint ctxAttribs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE};
    eglContext = eglCreateContext(eglDisplay, eglConfig, EGL_NO_CONTEXT, ctxAttribs);
    if (eglContext == EGL_NO_CONTEXT)
    {
        std::cerr << "Failed to create EGL context" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 7. Make context current
    if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext))
    {
        std::cerr << "Failed to make EGL context current" << std::endl;
        exit(EXIT_FAILURE);
    }

    // OpenGL state setup
    glClearColor(0.96f, 0.96f, 0.86f, 1.0f); // Beige background

    projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 10000.0f);

    // float dist = 12.0f;
    //  view = glm::lookAt(glm::vec3(dist, dist, dist),  // Camera position
    //                     glm::vec3(0.0f, 0.0f, 0.0f),  // Look-at point
    //                     glm::vec3(0.0f, 1.0f, 0.0f)); // Up vector
    view = glm::lookAt(glm::vec3(viewSettings[0], viewSettings[1], viewSettings[2]),  // Camera position
                       glm::vec3(viewSettings[3], viewSettings[4], viewSettings[5]),  // Look-at point
                       glm::vec3(viewSettings[6], viewSettings[7], viewSettings[8])); // Up vector

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

void ModelGl::ProcessFrame(int width, int height, char *filename)
{
    std::cout << "Processing frame... <------------------------" << std::endl;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // glDisable(GL_CULL_FACE);

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

    GLuint depthRenderbuffer;
    glGenRenderbuffers(1, &depthRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);

    // Create a texture to store the rendered image
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
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
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Pass the matrices to the shader
    glUseProgram(textureShaderProgram);
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(modelViewLoc, 1, GL_FALSE, glm::value_ptr(view));

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
    std::vector<unsigned char> pixels(width * height * 3); // RGB format
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Save the image using stb_image_write or another library
    stbi_flip_vertically_on_write(1); // Flip the image vertically
    if (!stbi_write_png(filename, width, height, 3, pixels.data(), width * 3))
    {
        std::cerr << "Failed to save image!" << std::endl;
    }
    else
    {
        std::cout << "Image saved as: " << filename << std::endl;
    }

    // Cleanup
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &framebuffer);
    glDeleteTextures(1, &texture);
    glDeleteRenderbuffers(1, &depthRenderbuffer);
}

int ModelGl::MakeObjectsOnGl()
{
    std::cout << "TODO: Add Objects without Textures." << std::endl;
    for (auto object : targetObjects)
    {
        std::cout << "Allocating OpenGL for target object. Num Facet :" << object->facets.size() << std::endl;
        object->AllocateGl();
    }

    for (auto object : fieldObjects)
    {
        std::cout << "Allocating OpenGL for field object. Num Facet :" << object->facets.size() << std::endl;
        object->AllocateGl();
    }

    return 0;
}

void ModelGl::FreeGl()
{
    for (auto object : targetObjects)
    {
        object->FreeGl();
    }

    for (auto object : fieldObjects)
    {
        object->FreeGl();
    }

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
}
