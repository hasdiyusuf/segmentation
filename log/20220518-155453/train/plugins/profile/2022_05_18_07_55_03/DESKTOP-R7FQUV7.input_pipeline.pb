	?B?i?q@?B?i?q@!?B?i?q@	???$0Q@???$0Q@!???$0Q@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?B?i?q@??ܵ?|??A6?>W[???Y?J?4??*	??????X@2F
Iterator::Model#??~j???!5)?J_D@)2??%䃞?17?B^??=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-???!?n??,=@)=?U?????1 ?N?>8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea2U0*???!@????P3@)???<,Ԋ?1?X??[*@:Preprocessing2U
Iterator::Model::ParallelMapV2'???????!??x?%@)'???????1??x?%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipGx$(??!???O??M@)?St$????1???b@? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?HP?x?!̣???@)?HP?x?1̣???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorn??t?!?v?'??@)n??t?1?v?'??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??_vO??!?I?U??5@)a2U0*?c?1@????P@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???$0Q@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ܵ?|????ܵ?|??!??ܵ?|??      ??!       "      ??!       *      ??!       2	6?>W[???6?>W[???!6?>W[???:      ??!       B      ??!       J	?J?4???J?4??!?J?4??R      ??!       Z	?J?4???J?4??!?J?4??JCPU_ONLYY???$0Q@b 