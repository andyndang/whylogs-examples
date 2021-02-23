plugins {
    java
    application
    id("com.github.davidmc24.gradle.plugin.avro") version "1.0.0"
}

application {
    mainClassName = "com.whylogs.examples.WhyLogsDemo"
}

group = "com.whylogs.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven(url = "https://packages.confluent.io/maven/")
}

dependencies {
    implementation("ai.whylabs:whylogs-core:0.0.2b3")
    implementation("org.apache.commons:commons-csv:1.8")
    implementation("org.apache.avro:avro:1.8.2")
    implementation("org.apache.kafka:kafka-clients:2.7.0")
    implementation("io.confluent:kafka-avro-serializer:6.1.0")
}


avro {
    isCreateSetters.set(true)
    isCreateOptionalGetters.set(false)
    isGettersReturnOptional.set(false)
    isOptionalGettersForNullableFieldsOnly.set(false)
    fieldVisibility.set("PUBLIC_DEPRECATED")
    outputCharacterEncoding.set("UTF-8")
    stringType.set("String")
    templateDirectory.set(null as String?)
    isEnableDecimalLogicalType.set(true)
}