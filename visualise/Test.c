#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

int main()
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