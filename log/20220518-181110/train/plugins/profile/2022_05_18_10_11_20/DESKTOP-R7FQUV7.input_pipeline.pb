	??H.??@??H.??@!??H.??@	??~??????~????!??~????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??H.??@??v????A??k	? @Y?/?$??*	?????X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7?[ A??!U??T?zA@)?<,Ԛ???1Z[JZ[J>@:Preprocessing2F
Iterator::Model?N@aã?!?O?OD@)B>?٬???1KZ[JZ[=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate? ?	???!?@??@?/@)???????1????(@:Preprocessing2U
Iterator::Model::ParallelMapV2?0?*??!?^?^%@)?0?*??1?^?^%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??j+????!U??T??M@)-C??6z?1W?W?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?O??nr?!??>?@);?O??nr?1??>?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!o?o?@)ŏ1w-!o?1o?o?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapr??????!:j\9j\2@)HP?s?b?1?N?N@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??~????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??v??????v????!??v????      ??!       "      ??!       *      ??!       2	??k	? @??k	? @!??k	? @:      ??!       B      ??!       J	?/?$???/?$??!?/?$??R      ??!       Z	?/?$???/?$??!?/?$??JCPU_ONLYY??~????b 