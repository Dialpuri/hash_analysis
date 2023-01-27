//
// Created by jordan on 1/24/23.
//

#ifndef HASH_ANALYSIS_VEC3_H
#define HASH_ANALYSIS_VEC3_H

struct Vec3 {

    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    float x = 0;
    float y = 0;
    float z = 0;

    Vec3& operator+= (const Vec3& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        this->z += rhs.z;
        return *this;
    }

    Vec3& operator/= (const int& rhs) {
        this->x /= rhs;
        this->y /= rhs;
        this->z /= rhs;
        return *this;
    }

    Vec3& operator- (const Vec3& rhs) {
        this->x -= rhs.x;
        this->y -= rhs.y;
        this->z -= rhs.z;
        return *this;
    }
};




#endif //HASH_ANALYSIS_VEC3_H
