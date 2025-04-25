
library(rtracklayer)
library(stringr)
setwd('~/disk2/deepTFBS/1.performance.evaluation/1.samples/')
BED <- import("./0.genome/chr1-5.bed", format = "BED")

bedDic <- "../../0.TF.narrowPeak/0.narrowPeak/"
TFList <- list.files(pattern = "narrowPeak", path = bedDic)
uniqueGene <- unique(unlist(str_extract_all(string = TFList, pattern = 'AT[1-5]G[0-9]+')))

labelMat <- matrix(0, nrow = length(BED), ncol = length(uniqueGene))
colnames(labelMat) <- uniqueGene

for(i in 1:length(TFList)){
  curGene <- str_extract(string = TFList[i], pattern = 'AT[1-5]G[0-9]+')
  narrowPeak <- import(paste0(bedDic, TFList[i]))
  res <- findOverlaps(query = BED, subject = narrowPeak, minoverlap = 100)
  labelMat[unique(queryHits(res)), curGene] <- 1
  cat(i, "\n")
}
labelSum <- apply(labelMat, 1, sum)

idx <- which(labelSum >= 5)
labelMat <- labelMat[idx, ]
write.table(labelMat, file = '3.raw_peak/samples_all_label.txt', sep = '\t', quote = F, 
            row.names = F, col.names = T)



#################Split samples###############
library(data.table)
BED <- as.data.frame(BED)
sampleName <- paste0(BED$seqnames, ":", BED$start, "-", BED$end)

labelMat <- as.matrix(labelMat)
class(labelMat) <- "numeric"
rownames(labelMat) <- sampleName[idx]


#################using chromosome 4 as testing samples
testIdx <- rownames(labelMat)[grep("chr4", rownames(labelMat))]
testLabel <- labelMat[testIdx, ]

remainIdx <- setdiff(rownames(labelMat), testIdx)
valIdx <- sample(remainIdx, 5000)
trainIdx <- setdiff(remainIdx, valIdx)

trainLabel <- labelMat[trainIdx, ]
valLabel <- labelMat[valIdx, ]

write.table(trainLabel, file = 'training_label.txt',
            sep = '\t', quote = F, row.names = F, col.names = F)
write.table(valLabel, file = 'validation_label.txt',
            sep = '\t', quote = F, row.names = F, col.names = F)
write.table(testLabel, file = 'test_label.txt',
            sep = '\t', quote = F, row.names = F, col.names = F)

write.table(trainIdx, file = 'training_IDs.txt',
            sep = '\t', quote = F, row.names = F, col.names = F)
write.table(valIdx, file = 'validation_IDs.txt',
            sep = '\t', quote = F, row.names = F, col.names = F)
write.table(testIdx, file = 'test_IDs.txt',
            sep = '\t', quote = F, row.names = F, col.names = F)
