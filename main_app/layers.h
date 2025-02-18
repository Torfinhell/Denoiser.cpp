#pragma once
#include"tensors.h"
struct Layer{
    public:
    virtual void forward(){}
    virtual ~Layer(){}
};
struct SimpleModel: public Layer{
    void forward() override;
    ~SimpleModel(){}
};