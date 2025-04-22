#include "ObjectGl.hpp"

int ObjectGl::ObjectGl(std::vector<Facet *> facets)
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

void ObjectGl::RenderObject(GLint textureUniformLoc)
{
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

    glBindVertexArray(vao);

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

        // glUseProgram(0);
        glDrawArrays(GL_TRIANGLES, facetCnt * 3, 3); // Render 3 vertices for the facet

        facetCnt++;

        GLenum err;
        err = glGetError();
        if (err != GL_NO_ERROR)
        {
            std::cerr << "OpenGL error: " << err << std::endl;
        }
    }
}

void ObjectGl::MakeTextureOnGl(
    std::vector<double *> &dev_facet_Pr,
    std::vector<double *> &dev_facet_Pi,
    float *dev_frag_stats)
{
    int numFacet = dev_facet_Pr.size();
    for (int i = 0; i < numFacet; ++i)
    {
        auto facet = facets[i];
        facet->WriteSurface(dev_facet_Pr[i], dev_facet_Pi[i], dev_frag_stats);
    }
}