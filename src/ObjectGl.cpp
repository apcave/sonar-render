#include "ObjectGl.hpp"

void ObjectGl::MakeVBO()
{
    std::cout << "Creating VBO for ObjectGl..." << std::endl;

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    if (glGenBuffers == nullptr)
    {
        std::cerr << "glGenBuffers is nullptr! Your context does not support VBOs." << std::endl;
    }
    // std::cout << "Extensions: " << glGetString(GL_EXTENSIONS) << std::endl;

    std::vector<float> vertexData;
    for (auto &facet : facets)
    {
        // Add vertex positions and texture coordinates for each facet
        vertexData.insert(vertexData.end(), {facet->v1.x, facet->v1.y, facet->v1.z, facet->texCoords[0], facet->texCoords[1],
                                             facet->v2.x, facet->v2.y, facet->v2.z, facet->texCoords[2], facet->texCoords[3],
                                             facet->v3.x, facet->v3.y, facet->v3.z, facet->texCoords[4], facet->texCoords[5]});
    }
    std::cout << "Test 1.01\n";
    // Upload the vertex data to the GPU
    glGenBuffers(1, &vbo);
    std::cout << "Test 1.2\n";
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    std::cout << "Test 1.3\n";
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);
    std::cout << "Test 1.4\n";
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    std::cout << "Test 2\n";

    // Vertex Attribute Object....
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    std::cout << "Test 3\n";

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    std::cout << "Test 3\n";
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    std::cout << "Test 5\n";
    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    std::cout << "Test 6\n";
    // Unbind the VBO and VAO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    std::cout << "Finshed VBO for ObjectGl..." << std::endl;
}

ObjectGl::~ObjectGl()
{
    std::cout << "Deleting ObjectGl..." << std::endl;

    // Delete the VBO
    if (vbo != 0)
    {
        std::cout << "2 Deleting ObjectGl..." << std::endl;
        glDeleteBuffers(1, &vbo);
        std::cout << "VBO " << vbo << " deleted." << std::endl;
        vbo = 0; // Reset to 0 to avoid dangling references
    }
    std::cout << "1 Deleting ObjectGl..." << std::endl;

    // Delete the VAO
    if (vao != 0)
    {
        glDeleteVertexArrays(1, &vao);
        std::cout << "VAO " << vao << " deleted." << std::endl;
        vao = 0; // Reset to 0 to avoid dangling references
    }

    std::cout << "Deleting facets..." << std::endl;
    for (auto facet : facets)
    {
        delete facet;
    }
    facets.clear();

    std::cout << "ObjectGl deleted." << std::endl;
}

ObjectGl::ObjectGl()
{
}

void ObjectGl::RenderObject(GLint textureUniformLoc)
{
    GLenum err;
    err = glGetError();
    if (err != GL_NO_ERROR)
    {
        std::cerr << "RenderObject OpenGL error: " << err << std::endl;
    }

    glBindVertexArray(vao);

    // Check if VAO is bound
    GLint boundVAO;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &boundVAO);
    if (boundVAO != vao)
    {
        std::cerr << "Error: VAO " << vao << " is not currently bound. Bound VAO: " << boundVAO << std::endl;
        return;
    }

    // PrintVBO();

    int facetCnt = 0;
    for (auto facGL : facets)
    {
        if (!glIsTexture(facGL->textureID))
        {
            std::cout << "Texture ID " << facGL->textureID << " is not valid." << std::endl;
        }

        // Bind the texture for this facet
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, facGL->textureID);
        glUniform1i(textureUniformLoc, 0);
        // facGL->PrintOpenGlTexture();
        // glUseProgram(0);
        glDrawArrays(GL_TRIANGLES, facetCnt * 3, 3); // Render 3 vertices for the facet

        facetCnt++;

        err = glGetError();
        if (err != GL_NO_ERROR)
        {
            std::cerr << "OpenGL error: " << err << std::endl;
        }
    }
}

void ObjectGl::AllocateGl()
{
    MakeVBO();
    for (auto facet : facets)
    {
        facet->AllocateGl();
    }
}

void ObjectGl::FreeGl()
{
    std::cout << "Deleting ObjectGl..." << std::endl;

    // Delete the VBO
    if (vbo != 0)
    {
        std::cout << "2 Deleting ObjectGl..." << std::endl;
        glDeleteBuffers(1, &vbo);
        std::cout << "VBO " << vbo << " deleted." << std::endl;
        vbo = 0; // Reset to 0 to avoid dangling references
    }
    std::cout << "1 Deleting ObjectGl..." << std::endl;

    // Delete the VAO
    if (vao != 0)
    {
        glDeleteVertexArrays(1, &vao);
        std::cout << "VAO " << vao << " deleted." << std::endl;
        vao = 0; // Reset to 0 to avoid dangling references
    }

    std::cout << "Deleting facets..." << std::endl;
    for (auto facet : facets)
    {
        facet->FreeGl();
    }
}

void ObjectGl::PrintVBO()
{
    std::cout << "VBO ID: " << vbo << std::endl;

    // Bind the VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Get the size of the buffer
    GLint bufferSize = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bufferSize);

    if (bufferSize > 0)
    {
        // Allocate a buffer to hold the data
        std::vector<float> vertexData(bufferSize / sizeof(float));

        // Read the data from the VBO
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, bufferSize, vertexData.data());

        // Print the data
        std::cout << "VBO Geometry Data:" << std::endl;
        for (size_t i = 0; i < vertexData.size(); i += 5) // Assuming 3 position + 2 texture coordinates
        {
            std::cout << "Vertex " << i / 5 << ": Position("
                      << vertexData[i] << ", " << vertexData[i + 1] << ", " << vertexData[i + 2] << "), "
                      << "TexCoords(" << vertexData[i + 3] << ", " << vertexData[i + 4] << ")" << std::endl;
        }
    }
    else
    {
        std::cout << "VBO is empty or not initialized." << std::endl;
    }

    // Unbind the VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}