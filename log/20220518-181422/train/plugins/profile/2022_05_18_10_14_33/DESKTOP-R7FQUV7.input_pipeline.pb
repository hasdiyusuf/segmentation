	Zd;?O	@Zd;?O	@!Zd;?O	@	?ǁiV?@?ǁiV?@!?ǁiV?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Zd;?O	@!?lV}??Az?):??@Y?5^?I??*dffff&Y@)      =2F
Iterator::Model?????K??!=???C?F@)aTR'????1?2Dֽ@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??j+????!!@<F?<@)?HP???1?Y0'%A8@:Preprocessing2U
Iterator::Model::ParallelMapV2F%u???!aZi>*@)F%u???1aZi>*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????Mb??!Ҽ?"$?/@)A??ǘ???1??h?M&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipK?46??!?GeO?bK@)ŏ1w-!?1?????7@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen??t?!?e??{@)n??t?1?e??{@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?O??nr?!6?wS??@);?O??nr?16?wS??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapΈ?????!'?Pg3}2@)??_?Le?1?m?
?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?ǁiV?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!?lV}??!?lV}??!!?lV}??      ??!       "      ??!       *      ??!       2	z?):??@z?):??@!z?):??@:      ??!       B      ??!       J	?5^?I???5^?I??!?5^?I??R      ??!       Z	?5^?I???5^?I??!?5^?I??JCPU_ONLYY?ǁiV?@b 