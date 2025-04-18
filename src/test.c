#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>

void PrintOpenGLErrors(char const *const Function, char const *const File, int const Line)
{
    bool Succeeded = true;

    GLenum Error = glGetError();
    if (Error != GL_NO_ERROR)
    {
        char const *ErrorString = (char const *)gluErrorString(Error);
        if (ErrorString)
            std::cerr << ("OpenGL Error in %s at line %d calling function %s: '%s'", File, Line, Function, ErrorString) << std::endl;
        else
            std::cerr << ("OpenGL Error in %s at line %d calling function %s: '%d 0x%X'", File, Line, Function, Error, Error) << std::endl;
    }
}

#ifdef _DEBUG
#define CheckedGLCall(x)                                         \
    do                                                           \
    {                                                            \
        PrintOpenGLErrors(">>BEFORE<< " #x, __FILE__, __LINE__); \
        (x);                                                     \
        PrintOpenGLErrors(#x, __FILE__, __LINE__);               \
    } while (0)
#define CheckedGLResult(x) \
    (x);                   \
    PrintOpenGLErrors(#x, __FILE__, __LINE__);
#define CheckExistingErrors(x) PrintOpenGLErrors(">>BEFORE<< " #x, __FILE__, __LINE__);
#else
#define CheckedGLCall(x) (x)
#define CheckExistingErrors(x)
#endif

void PrintShaderInfoLog(GLint const Shader)
{
    int InfoLogLength = 0;
    int CharsWritten = 0;

    glGetShaderiv(Shader, GL_INFO_LOG_LENGTH, &InfoLogLength);

    if (InfoLogLength > 0)
    {
        GLchar *InfoLog = new GLchar[InfoLogLength];
        glGetShaderInfoLog(Shader, InfoLogLength, &CharsWritten, InfoLog);
        std::cout << "Shader Info Log:" << std::endl
                  << InfoLog << std::endl;
        delete[] InfoLog;
    }
}

int main()
{
    GLFWwindow *window;

    if (!glfwInit())
        return -1;

    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        glfwTerminate();
        return -1;
    }

    while (!glfwWindowShouldClose(window))
    {
        CheckedGLCall(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        glBegin(GL_TRIANGLES);
        glColor3f(1.0f, 0.0f, 0.0f); // Red
        glVertex3f(-0.5f, -0.5f, 0.0f);
        glColor3f(0.0f, 1.0f, 0.0f); // Green
        glVertex3f(0.5f, -0.5f, 0.0f);
        glColor3f(0.0f, 0.0f, 1.0f); // Blue
        glVertex3f(0.0f, 0.5f, 0.0f);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}