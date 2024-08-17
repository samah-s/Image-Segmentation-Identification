CREATE SEQUENCE image_metadata_seq
START WITH 1
INCREMENT BY 1;


CREATE TABLE image_metadata (
    id NUMBER PRIMARY KEY,
    unique_id VARCHAR2(255) NOT NULL,
    master_id VARCHAR2(255) NOT NULL,
    image BLOB,
    x1 NUMBER NOT NULL,
    y1 NUMBER NOT NULL,
    x2 NUMBER NOT NULL,
    y2 NUMBER NOT NULL,
    confidence NUMBER NOT NULL,
    class NUMBER NOT NULL
);

CREATE OR REPLACE TRIGGER before_insert_image_metadata
BEFORE INSERT ON image_metadata
FOR EACH ROW
BEGIN
  IF :NEW.id IS NULL THEN
    SELECT image_metadata_seq.NEXTVAL INTO :NEW.id FROM dual;
  END IF;
END;

/